"""A self-written corpus with known ground truth, for measuring retrieval.

Why a purpose-built corpus instead of a downloaded one: the point of the
evaluation is to measure **this** pipeline (chunker + embedding model + vector
search), and to do that the correct answer for every question has to be known
exactly. Ten pages on ten clearly separable topics gives that, and the questions
are phrased in different words from the page they belong to, so a hit requires
semantic matching rather than keyword overlap.

What it is not: a benchmark. Ten synthetic pages cannot claim to represent real
textbooks. It is a regression test with an honest number attached.

Point the evaluation at your own PDF instead by passing ``--pdf`` and
``--questions``.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.pdf_factory import build_pdf

#: Ten pages, ten topics. Each is long enough to be split into several chunks by
#: the default 800-character chunker, so retrieval is exercised on the same
#: granularity as production.
PAGES: tuple[str, ...] = (
    (
        "Chapter 1: HTTP caching and conditional requests. A browser keeps a private cache of "
        "responses it has already received. When the cached copy is stale, the browser does not "
        "immediately re-download the representation. It sends a conditional request that carries a "
        "validator, usually the ETag the server supplied earlier, inside an If-None-Match header. "
        "The server compares that validator with the current one for the resource. If they match, "
        "the stored copy is still current, and the server replies with status 304 Not Modified and "
        "no body at all. The browser then reuses what it already has. A Last-Modified timestamp can "
        "play the same role through the If-Modified-Since header, although a timestamp is coarser "
        "than an entity tag because one second can contain several edits. This mechanism is what "
        "saves bandwidth on repeated visits: the client still asks, but nothing large is "
        "transferred back. Freshness is controlled separately by Cache-Control directives such as "
        "max-age, which let the client reuse a copy without contacting the origin at all."
    ),
    (
        "Chapter 2: Indexes and lookup cost. A table without an index forces the database to read "
        "every row to answer a query, which grows in proportion to the number of rows. A B-tree "
        "index replaces that scan with a descent from a root node to a leaf. Each node holds many "
        "keys and pointers, so a tree covering a million rows is only three or four levels deep, and "
        "each level costs one page read. The bound is therefore logarithmic in the row count: "
        "doubling the table adds one level at most, it does not double the work. Because the keys "
        "inside a node are sorted and neighbouring leaves are linked, the same structure also "
        "answers range queries efficiently. The trade is paid on writes. Every insert, update or "
        "delete has to keep the tree balanced, which costs extra page writes and can split a node "
        "that has filled up. That is why a table with many secondary indexes accepts writes more "
        "slowly than a table with none."
    ),
    (
        "Chapter 3: Hash tables and the load factor. A hash table stores each entry in a bucket "
        "chosen by hashing its key, which makes a lookup constant time on average. The load factor "
        "is the ratio of stored entries to buckets, and it is the number that decides how well the "
        "structure behaves. As the load factor rises, distinct keys are pushed into the same "
        "bucket, and each collision has to be resolved. Separate chaining stores a linked list per "
        "bucket, so a badly overloaded table degrades into a linear scan of one long list. Open "
        "addressing instead probes for the next free slot, which keeps the data in one array but "
        "creates clusters of occupied slots that grow and merge. Either way the average probe count "
        "climbs sharply once the table is more than about three quarters full. Production "
        "implementations therefore keep the load factor under a fixed bound and resize, allocating "
        "a larger array and rehashing every entry when the threshold is crossed."
    ),
    (
        "Chapter 4: Finding unreachable objects. Reference counting frees an object the moment its "
        "last pointer disappears, but it cannot collect a cycle, because two objects that point at "
        "each other each still have a non-zero count. Tracing collectors solve that by starting "
        "from the roots, the stack, the registers and the globals, and following every pointer "
        "outwards. Anything the traversal reaches is live. Everything else is unreachable and can "
        "be reclaimed. The simplest form is mark and sweep: mark walks the object graph and paints "
        "each reachable object, then sweep walks the heap linearly and frees whatever is still "
        "unpainted. The price is that the program has to be stopped while the graph is walked, "
        "which is why modern runtimes split the heap into generations and collect the young "
        "objects, where most garbage is, far more often than the old ones. Copying collectors go "
        "further and move the survivors into a second space, which also compacts the heap."
    ),
    (
        "Chapter 5: Preventing a fast sender from drowning a slow receiver. The two ends of a "
        "connection do not share a clock, and the receiving application may read its buffer more "
        "slowly than the network delivers data. Flow control exists to stop the sender from "
        "overrunning that buffer. The receiver advertises a window, the number of bytes it is "
        "currently willing to accept, and the sender may not have more unacknowledged data in "
        "flight than that figure allows. As the receiving application drains its buffer, the "
        "advertised window grows again and the sender is permitted to continue; if the buffer "
        "fills completely the window shrinks to zero and the sender pauses, probing "
        "periodically so that a lost update cannot deadlock the connection. Congestion control is "
        "a separate limit that estimates how much the path itself can carry, and the sender is "
        "held to the smaller of the two numbers."
    ),
    (
        "Chapter 6: Taking a failed backend out of rotation. A load balancer distributes "
        "connections across several servers so that no single one is overwhelmed. That only works "
        "while it knows which servers are healthy. It therefore probes each backend on a fixed "
        "interval, either by opening a connection, by reading a status endpoint, or by inspecting "
        "the response to an ordinary request. A backend that misses several consecutive probes is "
        "marked unhealthy and removed from the pool, so new connections go only to the survivors "
        "and users stop being sent to a machine that cannot answer. The check runs continuously "
        "because a server can fail long after it has passed its start-up test. When the failing "
        "backend starts answering again, it has to pass a threshold of successful probes before it "
        "is returned to rotation, which prevents a machine that is flapping from receiving a share "
        "of the traffic. Distribution itself may be round robin, weighted, or biased towards "
        "whichever healthy server has the fewest active connections."
    ),
    (
        "Chapter 7: Step size. Training a model means repeatedly nudging its parameters in the "
        "direction that reduces the error. The size of that nudge is the learning rate, and it "
        "matters more than almost any other setting. Too small a value makes progress so slow that "
        "the run appears to stall, and it can settle in a shallow region that a larger step would "
        "have escaped. Too large a value overshoots the minimum: the loss stops falling, then "
        "oscillates between opposite sides of the valley, and finally grows without bound as each "
        "step lands further from the target than the last. The usual remedy is to start larger than "
        "the final value and reduce it on a schedule, for example halving it when the validation "
        "loss stops improving, so the run moves quickly while the gradient is informative and "
        "settles finely at the end."
    ),
    (
        "Chapter 8: Surviving a power cut. A database has to keep its promises about committed "
        "transactions even when the machine loses power mid-write. Writing the modified pages "
        "directly to their final location cannot provide that, because a crash part way through "
        "leaves a page that is neither the old version nor the new one. Write-ahead logging fixes "
        "this by changing the order of operations: the description of the change is appended to a "
        "log and flushed to durable storage before any page is updated in place. A transaction is "
        "declared committed only once its log record is safely on disk. After a restart the "
        "recovery pass reads the log and replays the changes of committed transactions, then undoes "
        "the partial changes of transactions that had not committed. The database is driven back to "
        "a consistent state that reflects exactly the transactions that were acknowledged. The "
        "mechanism costs an extra write per transaction, which is why synchronous durability is "
        "measurably slower than letting the operating system buffer the log."
    ),
    (
        "Chapter 9: How a certificate becomes trustworthy. Encryption hides a conversation, but it "
        "does not by itself prove who is on the other end, since an attacker can present a key of "
        "their own. A certificate binds a public key to a name, and it is signed by an issuer. "
        "Trust starts from a small set of root certificates that ship with the operating system or "
        "browser. An intermediate issuer is signed by a root, the site's certificate is signed by "
        "that intermediate, and each signature can be verified with the public key of the issuer "
        "above it, so the chain leads back to something already trusted. The client also checks "
        "that the name it asked for appears in the certificate, that the validity dates cover the "
        "present moment, and that the certificate has not been revoked. A mismatch on any of those "
        "points aborts the connection with a warning rather than proceeding quietly."
    ),
    (
        "Chapter 10: Meaning as coordinates. A word-level index can only match the words that "
        "actually appear, so a search for one phrasing misses a document that expresses the same "
        "idea differently. An embedding model instead maps a piece of text to a fixed-length "
        "vector, trained so that passages used in similar contexts end up close together in that "
        "space. Two sentences can then be judged alike without sharing a single word, because "
        "closeness is measured by the angle between their vectors rather than by the overlap of "
        "their terms. Retrieval becomes a nearest-neighbour problem: embed the question with the "
        "same model that embedded the documents, then take the stored vectors with the highest "
        "cosine similarity. The model used at query time has to be the one that built the index, "
        "because vectors from two different models occupy different spaces and their distances are "
        "not comparable."
    ),
)


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """One question and the page that should answer it.

    ``expected_page`` of ``None`` marks a question the corpus deliberately cannot
    answer, which tests that the pipeline abstains instead of retrieving noise.
    """

    question: str
    expected_page: int | None


#: 1-based page numbers, matching what the README calls a page and what the API
#: returns as ``sources[].page``.
QUESTIONS: tuple[EvalQuestion, ...] = (
    EvalQuestion(
        "How can a browser avoid downloading a page again when the copy it already "
        "holds is still current?",
        1,
    ),
    EvalQuestion(
        "Why does looking up a single row stay fast when a table holds millions of them?",
        2,
    ),
    EvalQuestion(
        "What goes wrong when a fixed array of buckets becomes too full?",
        3,
    ),
    EvalQuestion(
        "How does a runtime identify objects that the running program can no longer reach?",
        4,
    ),
    EvalQuestion(
        "What stops a fast sender from overwhelming a receiver that reads slowly?",
        5,
    ),
    EvalQuestion(
        "What does a dispatcher do when one of its backends stops answering probes?",
        6,
    ),
    EvalQuestion(
        "What happens if every training step moves the parameters too far?",
        7,
    ),
    EvalQuestion(
        "How is acknowledged work recovered after the power suddenly fails?",
        8,
    ),
    EvalQuestion(
        "Who vouches for a website's public key, and how does the client come to trust it?",
        9,
    ),
    EvalQuestion(
        "Why can two passages count as similar when they share no words at all?",
        10,
    ),
    # Out-of-corpus: nothing above the similarity floor should be retrieved.
    EvalQuestion("What is the boiling point of mercury in degrees Celsius?", None),
    EvalQuestion("How do I replace the inner tube of a bicycle tyre?", None),
    EvalQuestion("Which country won the football world cup in 2018?", None),
)


def build_corpus_pdf() -> bytes:
    """Render the corpus to a real PDF, so the evaluation runs the real parser."""
    return build_pdf(list(PAGES))
