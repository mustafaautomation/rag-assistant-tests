# Evaluation sets

Both files were **hand-written and hand-labelled** for this repository by reading the articles
in `corpus/`. They are small (42 retrieval questions, 22 answer questions) and are meant as
regression gates for this demo, not as a benchmark.

- `retrieval_set.jsonl`: a question and the article ids (`relevant`) that answer it. Questions
  are phrased the way a rider might ask, not copied from the articles.
- `answer_set.jsonl`: answerable questions with strings the answer must contain, and
  unanswerable questions (`answerable: false`) where the assistant must refuse.
- `thresholds.json`: the CI gate. Values were set slightly below the measured baseline of the
  default offline pipeline; see the README for the measured numbers.
