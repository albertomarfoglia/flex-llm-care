"""
FAISS-based vector store for FHIR resource retrieval.
"""

import hashlib
import json
import os
import string
import warnings
from pathlib import Path

from huggingface_hub import login
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

LABELS = list(string.ascii_uppercase)
"""Candidate labels shown to the LLM (``A)`` … ``Z)``); bounds ``k`` to 26."""

INDEX_META_FILE = "index_meta.json"


def _default_device() -> str:
    # pylint: disable=import-outside-toplevel
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class VectorStore:
    """
    Builds and queries a FAISS vectorstore from serialised resource documents.

    Parameters
    ----------
    embeddings : str | Embeddings
        HuggingFace model name for the embedding model, or a ready LangChain
        ``Embeddings`` instance (useful for tests).
    docs : list[str]
        JSON-serialised resource points as returned by ``Loader.docs``.
    device : str | None, optional
        Device for the embedding model; ``None`` (default) picks CUDA if
        available, else CPU.
    normalize : bool, optional
        Whether to normalize embeddings, by default ``False``.
    revision : str | None, optional
        Embedding model revision (commit hash) to load.
    """

    __LOGGED_IN = False

    def __init__(
        self,
        embeddings: str | Embeddings,
        docs:       list[str],
        device:     str | None = None,
        normalize:  bool = False,
        revision:   str | None = None,
    ):
        self.__docs       = docs
        self.__meta       = {
            "embeddings": embeddings if isinstance(embeddings, str) else type(embeddings).__name__,
            "revision":   revision,
            "normalize":  normalize,
            "docs_md5":   hashlib.md5(json.dumps(docs).encode(), usedforsecurity=False).hexdigest(),
        }
        self.__embeddings = (
            embeddings if isinstance(embeddings, Embeddings)
            else self.__load_embeddings(embeddings, device or _default_device(), normalize, revision)
        )
        self.__db: FAISS | None = None

    @classmethod
    def __ensure_hf_login(cls) -> None:
        """Login to HuggingFace Hub once per process.

        If ``HF_KEY`` is not set, login is skipped and only public
        (non-gated) models can be downloaded.
        """
        if not cls.__LOGGED_IN:
            token = os.environ.get("HF_KEY")
            if not token:
                warnings.warn(
                    "HF_KEY not set: skipping HuggingFace login. "
                    "Gated models (e.g. meta-llama/*) will not be accessible."
                )
                return
            login(token=token)
            cls.__LOGGED_IN = True

    def __load_embeddings(
        self,
        model:     str,
        device:    str,
        normalize: bool,
        revision:  str | None,
    ) -> HuggingFaceEmbeddings:
        self.__ensure_hf_login()
        model_kwargs = {"device": device}
        if revision:
            model_kwargs["revision"] = revision
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": normalize},
        )

    @staticmethod
    def __iter_chunks(items: list[str], size: int):
        for i in range(0, len(items), size):
            yield items[i:i + size]

    @staticmethod
    def __doc_to_chunks(doc: str, chunk_size: int) -> list[str]:
        parsed   = json.loads(doc)
        resource = parsed["resource"]

        def make_chunk(content: str) -> str:
            return json.dumps({"resource": resource, "content": content})

        chunked_keys = ("examples", "documentation")
        return [
            make_chunk("\n".join(group))
            for key, value in parsed.items()
            if key != "resource"
            for group in (
                VectorStore.__iter_chunks(value, chunk_size)
                if key in chunked_keys
                else [[value]]
            )
        ]

    def __build_chunks(self, chunk_size: int) -> list[str]:
        return [
            chunk
            for doc in self.__docs
            for chunk in VectorStore.__doc_to_chunks(doc, chunk_size)
        ]

    @property
    def db(self) -> FAISS:
        """
        Return the FAISS index.

        Raises
        ------
        RuntimeError
            If the index has not been built or loaded yet.
        """
        if self.__db is None:
            raise RuntimeError(
                "Vectorstore not ready. Call build() or load() first."
            )
        return self.__db

    @property
    def labels(self) -> list[str]:
        """Labels assigned, in order, to the retrieved candidates."""
        return list(LABELS)

    def __index_meta(self, chunk_size: int) -> dict:
        return {**self.__meta, "chunk_size": chunk_size}

    def build(self, chunk_size: int = 6, save_path: Path | None = None) -> "VectorStore":
        """
        Build the FAISS index from the documents passed at construction time.

        Parameters
        ----------
        chunk_size : int, optional
            Number of items per chunk for ``examples`` and ``documentation``
            fields, by default ``6``.
        save_path : Path, optional
            If provided, the index is persisted to disk at this path, together
            with ``index_meta.json`` describing how it was built.
        """
        docs     = [Document(chunk) for chunk in self.__build_chunks(chunk_size)]
        self.__db = FAISS.from_documents(docs, self.__embeddings)
        if save_path:
            self.__db.save_local(str(save_path))
            (Path(save_path) / INDEX_META_FILE).write_text(
                json.dumps(self.__index_meta(chunk_size), indent=2), encoding="utf-8")
        return self

    def load(self, load_path: Path) -> "VectorStore":
        """
        Load a previously saved FAISS index from disk.

        Parameters
        ----------
        load_path : Path
            Directory where the FAISS index was saved.

        Raises
        ------
        RuntimeError
            If no index is found at ``load_path``.
        """
        if not load_path.exists():
            raise RuntimeError(
                f"No index found at {load_path}. Call build() first."
            )
        self.__db = FAISS.load_local(
            str(load_path),
            self.__embeddings,
            allow_dangerous_deserialization=True,
        )
        return self

    def load_or_build(self, path: Path, chunk_size: int = 6) -> "VectorStore":
        """
        Load the index at ``path`` if it was built from the same documents,
        embedding model and chunk size; otherwise (re)build and save it.
        """
        meta_fp = Path(path) / INDEX_META_FILE
        if meta_fp.exists() and json.loads(meta_fp.read_text()) == self.__index_meta(chunk_size):
            return self.load(Path(path))
        if Path(path).exists():
            warnings.warn(f"Index at {path} is stale or has no metadata; rebuilding it.")
        return self.build(chunk_size=chunk_size, save_path=Path(path))

    def get_context(
        self,
        query:                       str,
        k:                           int  = 10,
        every_represented_candidate: bool = False,
    ) -> dict:
        """
        Retrieve the top-k most relevant FHIR resource chunks for a query.

        Parameters
        ----------
        query : str
            Free-text clinical description to search against the vector store.
        k : int, optional
            Number of results to return (1–26), by default ``10``.
        every_represented_candidate : bool, optional
            If ``True``, returns at most one chunk per resource type — i.e.
            the top-k results cover k distinct FHIR resources.
            If ``False``, returns the top-k chunks regardless of resource type,
            so the same resource may appear multiple times.

        Returns
        -------
        dict
            - ``context``   (str)       : Labelled chunks formatted as
            ``"<LABEL>) <Resource>: <content>"`` and joined by newline,
            ready to be injected into the prompt template.
            - ``resources`` (list[str]) : Ordered list of matched resource
            names (lowercase), in the same order as ``context``.

        Raises
        ------
        ValueError
            If ``k`` is outside 1–26.
        RuntimeError
            If the vector store has not been built or loaded yet.
        """
        if not 1 <= k <= len(LABELS):
            raise ValueError(f"k must be between 1 and {len(LABELS)}, got {k}")
        n_docs  = self.db.index.ntotal if every_represented_candidate else k
        results = self.db.similarity_search_with_score(query=query, k=n_docs)

        output, resources = [], []
        for doc, _ in results:
            page = json.loads(doc.page_content)
            res  = page["resource"]
            if every_represented_candidate and res.lower() in resources:
                continue
            if len(resources) >= k:
                break
            resources.append(res.lower())
            output.append(f"{LABELS[len(output)]}) {res}: {page['content']}")

        return {"context": "\n".join(output), "resources": resources}
