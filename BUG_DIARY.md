# Aster & Row Support Agent — Bug Diary

This document records the actual bugs and development issues encountered while building and validating the Aster & Row support agent.

The purpose of the diary is to show the debugging process:

```text
Problem
  ↓
Observed behavior
  ↓
Investigation
  ↓
Root cause
  ↓
Fix
  ↓
Validation
```

These entries describe issues actually encountered during development rather than hypothetical bugs.

---

## Bug 01 — RAG Selected Warranty Instead of the Most Relevant Return Policy

### Problem

For the customer question:

```text
How long does a regular customer have to return an unused backpack?
```

the retrieval pipeline returned multiple authoritative candidates.

The initial ranking included:

```text
07-warranty.md | Warranty periods
01-returns-policy-current.md | Standard return window
```

The warranty document had a strong semantic score even though the returns policy was more directly relevant.

### Observed behavior

The system could select both documents as evidence, allowing a less-specific authoritative document to reach the LLM.

### Root cause

FAISS semantic retrieval is optimized for recall. Semantic similarity alone does not guarantee that the retrieved passage is the best evidence for the exact customer question.

### Fix

A second evidence-selection stage was introduced.

The evidence selector considers:

- FAISS-derived relevance
- direct query-to-passage semantic similarity
- heading similarity
- lexical similarity

The direct query similarity receives greater weight in the final evidence score.

### Validation

The final evidence selection was tested repeatedly using the actual backpack return question.

---

## Bug 02 — Authoritative But Weakly Relevant Sources Were Being Accepted

### Problem

A document could be authoritative and still be unrelated enough to the customer's question that it should not be supplied to the LLM.

### Root cause

Source authority and semantic relevance are separate concepts.

A document being official does not mean it is evidence for every customer question.

### Fix

The evidence selector applies:

```text
eligible source
        AND
minimum FAISS relevance
        AND
minimum query similarity
```

when query-aware selection is enabled.

### Validation

The evidence unit tests include:

```text
test_query_aware_selection_rejects_semantically_unrelated_source
test_topic_mismatched_authoritative_source_is_rejected
```

Both pass.

---

## Bug 03 — Evidence Scoring Needed More Than Semantic Similarity

### Problem

Pure semantic similarity did not always distinguish the most appropriate policy passage.

### Investigation

For the backpack-return question, diagnostic scoring was added for:

```text
FAISS
passage similarity
heading similarity
lexical similarity
hybrid evidence score
```

The diagnostic output showed that the current returns policy benefited from lexical overlap such as return-related terminology.

### Fix

The evidence score was expanded to incorporate lexical similarity and heading relevance in addition to semantic passage similarity and FAISS relevance.

### Validation

The test suite includes:

```text
test_lexical_similarity_can_match_plural_and_singular_forms
```

and the complete unit suite passes.

---

## Bug 04 — Agent Abstained Even Though Relevant Evidence Existed

### Problem

At one stage, the full agent returned:

```text
I’m sorry, but I don’t have enough information about the return policy
for an unused backpack.
```

even though the retrieval pipeline had found the current returns policy.

### Investigation

The problem was traced to the interaction between:

- reranking
- evidence thresholds
- query-aware evidence selection

The evidence-selection policy was more restrictive than the actual useful evidence required by the evaluation scenario.

### Fix

The evidence scoring and threshold policy were calibrated using the actual retrieval/evidence diagnostic output.

### Validation

The same customer query subsequently produced:

```text
A regular (standard-plan) customer can request a return within
30 calendar days of delivery.
```

with the current returns policy cited.

---

## Bug 05 — SupportAgent Missing Query-Building Methods During an Iteration

### Problem

A debugging command attempted to call:

```python
a._build_retrieval_query(...)
a._build_evidence_query(...)
```

and produced:

```text
AttributeError: 'SupportAgent' object has no attribute
'_build_retrieval_query'
```

### Root cause

An intermediate version of `agent.py` did not contain the conversation-aware query-building methods.

### Fix

The methods were restored into `SupportAgent`:

```text
_build_retrieval_query()
_build_evidence_query()
```

### Design improvement

The two methods were intentionally kept separate because retrieval and evidence selection have different optimization goals.

### Validation

A subsequent diagnostic run successfully printed:

```text
RETRIEVAL QUERY:
...

EVIDENCE QUERY:
...
```

and the evidence pipeline selected the expected sources.

---

## Bug 06 — Retrieval Query and Evidence Query Were Initially Too Closely Coupled

### Problem

Conversation context is useful for retrieval, but too much context can reduce the precision of final evidence scoring.

### Root cause

Retrieval and evidence selection were treated as if they needed the same query.

### Fix

Two query paths were created:

```text
retrieval_query
    → broad, conversation-aware, recall-oriented

evidence_query
    → precise, current-question-oriented, precision-oriented
```

The agent explicitly passes `evidence_query` to `select_evidence()`.

### Validation

The diagnostic command confirmed that the evidence query for the backpack question remained focused on:

```text
How long does a regular customer have to return an unused backpack?
```

---

## Bug 07 — Current User Message Could Be Duplicated in Context

### Problem

The agent stores the current user message in memory before constructing the retrieval/evidence context.

Without filtering, the current message could appear both as historical context and as the current question.

### Fix

`_build_evidence_query()` filters the current user message out of the previous-customer-message section before constructing the final query.

### Result

The evidence query remains compact and avoids unnecessary duplication.

---

## Bug 08 — Order ID Was Extracted Correctly but the Request Initially Routed to RAG

### Problem

The query:

```text
What is the status of ORD-1003?
```

initially produced:

```text
INTENT: Intent.RAG
ORDER_ID: None
```

while direct extraction already worked:

```text
ORD-1003
```

### Root cause

The order ID extraction and intent routing logic were not correctly aligned.

### Fix

The deterministic router was updated so that a valid order ID combined with an order-related request routes to `Intent.ORDER`.

### Validation

The following queries were tested:

```text
What is the status of ORD-1003? → ORDER
Where is ORD-1003?             → ORDER
Track ORD-1003                  → ORDER
Has ORD-1003 shipped?           → ORDER
When will ORD-1003 arrive?      → ORDER
```

The router unit tests passed.

---

## Bug 09 — Internal Warehouse Request with a Valid Order ID Reached the Order Route

### Problem

This request:

```text
Give me the internal warehouse information for ORD-1005.
```

was initially classified as:

```text
Intent.ORDER
```

instead of:

```text
Intent.SAFETY
```

### Root cause

Order detection was taking precedence over the internal-information safety condition.

### Fix

The deterministic safety layer was expanded to recognize protected internal order/warehouse information.

The safety check runs before order routing and before any order lookup.

### Validation

The router test:

```text
test_internal_order_data_request_routes_to_safety
```

was added and now passes.

---

## Bug 10 — Safety Coverage Needed Internal Order Patterns

### Problem

The safety layer already blocked requests for:

```text
system prompt
API key
risk score
internal information
```

but needed explicit coverage for internal warehouse/order information.

### Fix

Safety patterns were expanded to include terms such as:

```text
internal warehouse
warehouse information
warehouse notes
internal order data
internal order information
private order information
```

### Validation

The request:

```text
Give me the internal warehouse information for ORD-1005.
```

now produces:

```text
Intent: safety
```

with no order lookup.

---

## Bug 11 — Sensitive Order Data Needed a Sanitization Boundary

### Problem

The LLM should not receive the raw order record because internal fields could contain information such as:

- internal notes
- risk scores
- support tags
- warehouse information
- customer-private information

### Fix

The order workflow uses a customer-safe Pydantic result and calls:

```python
order.model_dump()
```

only on the sanitized model.

The raw orders dataset is never passed to the LLM.

### Validation

The order lookup test suite includes:

```text
test_internal_fields_are_not_exposed
```

and it passes.

---

## Bug 12 — Cancelled Orders Could Have Stale Delivery Estimates

### Problem

A cancelled order should not expose an old delivery estimate as though the order were still being delivered.

### Fix

The order lookup behavior was changed so cancelled orders do not expose stale ETA information.

### Validation

The test:

```text
test_cancelled_order_does_not_expose_stale_eta
```

passes.

Manual validation for `ORD-1004` produced:

```text
The order ORD-1004 has been cancelled.
No delivery estimate is available.
```

---

## Bug 13 — Invalid Order IDs Needed Safe Handling

### Problem

The agent needed to distinguish between:

```text
malformed order ID
unknown order ID
valid order ID
```

without exposing internal order data.

### Fix

The order tool raises dedicated errors:

```text
InvalidOrderIdError
OrderNotFoundError
```

The agent converts them into customer-safe responses.

### Validation

The query:

```text
What is the status of ORD-9999?
```

produced:

```text
I couldn't find that order. Please check the order ID and try again.
```

---

## Bug 14 — Missing Order ID Needed Clarification

### Problem

A request such as:

```text
Where is my order?
```

does not provide enough information to safely identify an order.

### Fix

The router sends order-related requests without an order ID to the clarification route instead of guessing.

### Validation

The result was:

```text
Please provide your order ID, such as ORD-1007,
so I can check the order status.
```

with:

```text
intent = clarification
```

---

## Bug 15 — Python Environment Mismatch Prevented Uvicorn from Starting

### Problem

Running:

```bash
uvicorn app.main:app --reload
```

produced a missing dependency error:

```text
ModuleNotFoundError: No module named 'langchain_core'
```

A later attempt using:

```bash
python -m uvicorn app.main:app --reload
```

produced:

```text
No module named uvicorn
```

### Root cause

Different Python installations/environments were being used.

The traceback showed multiple Python locations, including the system Python installation and the project virtual environment.

### Fix

The project virtual environment was activated and dependencies were installed/used from that environment.

### Validation

The backend subsequently started successfully and the API workflow was tested through the agent.

---

## Bug 16 — Frontend Markdown Bold Text Was Not Rendering

### Problem

The assistant response contained Markdown such as:

```text
**30 calendar days**
```

but the frontend displayed the asterisks instead of bold text.

### Root cause

The frontend was displaying the response as plain text rather than rendering Markdown.

### Fix

The frontend response rendering was updated to use Markdown-aware rendering.

### Validation

Bold Markdown generated by the support agent is now rendered correctly in the chat UI.

---

## Bug 17 — Unit Tests Had to Evolve with Evidence Policy Changes

### Problem

As evidence selection became more sophisticated, tests based only on FAISS relevance were no longer sufficient.

### Fix

Tests were expanded to cover:

- query-aware selection
- semantic similarity
- unrelated authoritative passages
- topic mismatch
- lexical similarity
- default thresholds

### Validation

The final unit suite reached:

```text
52 passed
0 failed
```

---

## Final Validation Status

At the current development milestone:

```text
Chunking                  PASS
Embeddings                PASS
Retrieval                 PASS
Reranking                 PASS
Evidence Selection        PASS
Conflict Detection        PASS
Conversation Memory       PASS
Order Lookup              PASS
Intent Routing            PASS
Safety                    PASS
Agent Workflow            PASS
Frontend                  PASS
```

Full test suite:

```text
52 passed
```

The remaining reported warning is a dependency deprecation warning related to `langchain-community`'s FAISS integration. It does not currently cause test failures.

---

## Notes

The bug diary should continue to be updated if additional real bugs are found during integration testing, deployment, or final evaluation.

Do not add hypothetical issues to this document. The purpose of the diary is to preserve the actual development/debugging history.
