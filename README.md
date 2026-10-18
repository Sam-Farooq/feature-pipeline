# feature-pipeline

Features declared once, computed two ways: in batch over a history to build a
training frame, and incrementally into Redis to be read at decision time. A
check runs the same declaration down both paths and compares the answers.

Training and serving skew is the failure it is built against. A model trained
on a number the serving layer computes slightly differently looks fine in
validation and is wrong in production, and nothing in the training run says
so.

## The four ways the two paths come apart

| How | What it looks like | What stops it here |
|---|---|---|
| The batch job sees data the serving layer could not have | validation improves, production does not move | every event carries `event_at` and `recorded_at`, and the batch filter uses both |
| One path's window edge moves | a count off by one on exactly the rows near the edge | one `within_window`, in a module with no pandas and no redis in it, called by both |
| A definition changes and old values keep being served | the number is plausible and means something else | the feature's hash is part of its Redis field name, so a redefined feature reads as absent rather than as the old value |
| A cache miss is filled with a default | "no transactions in 24h" and "we have never heard of this account" become the same input | a miss returns no value and says so; the empty-window value is only ever produced by a computation |

The first one is the expensive one, so it gets its own section.

## Recorded at, not just happened at

One account, one day, and the same five features at two decision times three
hours apart. All of it comes out of `examples/events.csv`, which is eleven
hand written transactions.

```
featpipe offline --view account_activity \
  --events examples/events.csv --spine examples/spine_accounts.csv
```

```
account_id                     as_of is_fraud  txn_count_24h  amount_sum_24h  amount_mean_7d  max_amount_7d  distinct_countries_7d
        A1 2026-03-02 09:00:00+00:00        0              2           85.75           89.15          200.0                      1
        A2 2026-03-02 09:00:00+00:00        0              3           110.0       36.666667           60.0                      1
        A3 2026-03-02 09:00:00+00:00        0              0             0.0            <NA>           <NA>                      0
        A1 2026-03-02 12:00:00+00:00        1              3          585.75         157.625          500.0                      2
```

A1 has a 500.00 transaction in Dubai, stamped 22:40 the previous night and
recorded at 11:15 the next morning, twelve hours and thirty five minutes
late. The 09:00 row
cannot see it: two transactions, 85.75, one country. The 12:00 row can: three,
585.75, two countries. Filtering on `event_at` alone would have put that
transaction, that amount and that second country into the 09:00 row, and the
model would have been trained on a fact that arrives two hours after it has to
decide.

Three rules, and the tests name them:

- `recorded_at <= as_of`. Recorded exactly at the decision time counts. A1's
  10.25 was recorded at 09:00:00.000 and is in the 09:00 row.
- the window's lower edge is exclusive and its upper edge is not, so a 24h
  window holds 24 hours and not 24 hours and one event. A1's 120.00 is stamped
  exactly 24h before the 09:00 label and is outside the 24h count, inside the 7d
  mean.
- an empty window is not a missing window. A3 has one transaction and it is
  older than both windows, so its count is 0, its sum is 0.0, and its mean is
  `<NA>`, because the mean of nothing is not zero.

That last rule is declared, not inferred. Every feature in a `.toml` carries
`empty_window = "zero"` or `"unknown"` and the loader refuses a definition that
omits it. Whether an empty 24h window means no transactions or an account
nobody has seen is a modelling decision, and `mean` and `max` may not declare
`"zero"` at all, because there is no average of nothing.

## One definition, two hashes

`definitions/*.toml` is read by both paths. `tomllib` has been in the standard
library since 3.11, so the format costs nothing.

```
featpipe defs
```

```
account_activity  key fp:account_activity:8acaee859b:<account_id>
  every 10m, stale after 20m, ttl 36h
  txn_count_24h          count          -         24h  empty=zero     field=txn_count_24h@bd98ecd1
  amount_sum_24h         sum            amount    24h  empty=zero     field=amount_sum_24h@6e9db537
  amount_mean_7d         mean           amount     7d  empty=unknown  field=amount_mean_7d@96570ff4
  max_amount_7d          max            amount     7d  empty=unknown  field=max_amount_7d@3a828893
  distinct_countries_7d  distinct_count country    7d  empty=zero     field=distinct_countries_7d@f74f9119
```

A row is a Redis hash. The key carries a hash of what makes a row a row: the
view name, the entity column and the two time columns. A field carries a hash
of one feature's aggregation, column, window and empty policy.

That split is the whole reason a backfill is possible:

- narrowing `txn_count_24h` to 12h renames one field. Serving looks for the new
  name, does not find it, and reports it absent. The old value sits there under
  its old name until the key expires, and is never handed to anyone as the new
  feature.
- adding a feature adds a field. The key does not move, the other fields do not
  move, and the ttl does not move.

Putting the feature set into the key instead would be simpler and would mean
every new feature invalidates every row, which is exactly the rebuild a
backfill exists to avoid.

## Fresh, stale, missing

A read returns one of three things, and collapsing any two of them is how skew
gets into production quietly.

| Status | Means | What the caller gets |
|---|---|---|
| FRESH | stored within the view's staleness budget | the values |
| STALE | real values the store has held too long, or whose age two clocks disagree about | the values and the age, and `require()` raises |
| MISS | no row, no field at the current definition version, or no way to date the row | no values at all, never substituted |

`account_activity` allows 20 minutes. A row read 1200 seconds after the store
took it is fresh; at 1201 it is stale and says `stored 1201s ago, budget is
1200s`. A partly written row is a miss rather than a row with holes, and a row
that cannot be dated is a miss too, because a value nobody can date cannot be
judged against a budget.

### Which clock the age is measured on

A row carries two timestamps and they answer different questions.

| Field | Written by | Means |
|---|---|---|
| `_as_of` | the writer | the instant the features were computed over. Event time, so the batch path joins on it and a backfill matches against it |
| `_written_at` | the store's own clock, read with `TIME` | when the store took the row, and the only one freshness is measured against |

Freshness used to be the reader's `now` minus `as_of`, which trusted the clock
of a machine the reader never talks to. A writer an hour fast stamped every row
an hour into the future: a row computed at 09:00 and read at 09:25 came back
FRESH against a twenty minute budget, with its age reported as -2100 seconds,
and nothing counted it. A slow writer fails the other way and makes a current
row look stale, which costs a recompute nobody needed.

Both ends of the subtraction come from the store's clock now, so neither the
writer's nor the reader's is in the arithmetic. `read_vector` has no `now`
argument: a caller cannot choose the clock its answer is judged on.
`featpipe serve --at` is still there and is an explicit override for looking at
a row, rather than the only way to ask.

What that costs, and it is worth knowing before trusting it:

- a write is one round trip more, because `TIME` has to be asked before the
  `HSET` its answer goes into and cannot ride inside the `MULTI`. A read of a
  row that exists is one more too, though a miss still costs one. A Lua script
  would get the write back and is not written.
- a row with no `_written_at`, or an unreadable one, is a MISS. That includes
  every row written by a version before this one, until the next
  materialisation replaces it, which for `account_activity` is ten minutes.
  The alternative is dating it by the writer's clock again.
- a store clock that steps backwards by more than the five second skew
  allowance is answered STALE with the reason, not FRESH off a negative age.

An `as_of` more than that allowance *ahead* of the store clock is refused at
the write, exit 2. The comparison has to be against a clock the writer does not
own: a writer checking its own `as_of` against its own `now()` detects nothing,
because a fast clock is fast in both halves of that comparison. The ninth step
of `featpipe demo` is that refusal.

The ttl is not the staleness budget. It is 36h, and the loader refuses any view
whose ttl is shorter than one materialisation interval plus the staleness
allowance, since those keys expire between runs and every read becomes a miss
for a reason nothing in the definition explains. One `as_of` per row, so every
field in a key was computed at the same instant or the write is refused.

## Running it

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
featpipe demo
```

`demo` runs the whole cycle in one process against `fakeredis`. No Redis, no
network, no service. Nine steps: materialise, read it fresh, read it 25
minutes later, read an entity the stream has never seen, read one whose history
has aged out, redefine a feature, add a new one and back it in, aim that
backfill at a row that has expired, and try to write a row stamped an hour
ahead of the store's clock.

Steps 4 and 5 are the pair worth reading:

```
4. an entity the stream has never seen: a miss, not a set of zeros
account_activity/A9  MISS  age -
  no row for this entity at this view version
  txn_count_24h          no value
  amount_sum_24h         no value

5. an entity whose history has all aged out: zeros where zero is the
   answer, unknown where there is nothing to average
account_activity/A3  FRESH  age 0s
  txn_count_24h          0
  amount_sum_24h         0.0
  amount_mean_7d         unknown
```

A3 kept a row after its last transaction fell out of the 7d window. The
accumulator holds an entry per entity it has ever seen and only the events of
the longest window, so an account with nothing recent still answers 0 rather
than going silent. The parity check is what forced that: dropping the entity
made serving answer MISS where the batch path answered 0.

`materialize` and `serve` are separate commands, and that is why `demo`
exists:

```
featpipe materialize --view account_activity --events examples/events.csv \
  --as-of 2026-03-02T09:00:00Z --redis memory
```

```
account_activity at 2026-03-02T09:00:00Z: 3 rows, 15 fields, 9 events ingested, 0 dropped as too old, 1 evicted
the memory client is per process, so nothing survives this command
```

`--redis` otherwise takes any `redis://` url, and then the two commands share
a server. On `serve`, `--strict` exits 6 unless the row is fresh, which is the
shape a readiness check wants. `featpipe parity` exits 5 on a disagreement,
and anything refused exits 2.

## The parity check

```bash
featpipe parity --view account_activity \
  --events examples/events.csv --spine examples/spine_accounts.csv
featpipe parity --view merchant_activity \
  --events examples/events.csv --spine examples/spine_merchants.csv
```

```
account_activity: 4 spine rows, 20 values compared, 0 disagreements
merchant_activity: 3 spine rows, 6 values compared, 0 disagreements
```

For every spine row it computes the batch value, builds the serving path's
accumulator out of the events recorded by that instant, writes the vector, and
reads it back through the real serving code. The comparison is against what a
caller would be handed, Redis encoding and freshness included, not against an
in-memory dict.

It is worth something only because the two sides are different code.
`featurepipe/offline/frames.py` is a vectorised pandas join over the whole
spine at once. `featurepipe/online/accumulator.py` is a row-at-a-time stream
consumer keeping sorted per-entity windows, with no pandas in it, and a test
parses its imports to keep it that way. Comparing an implementation against
itself proves nothing.

A check that cannot fail is worth as little, so `tests/test_parity.py`
reintroduces each known drift on purpose and requires the check to catch it:
an inclusive lower edge in the serving path alone (A1's count goes from 2 to 3),
an empty window answered with 0.0 instead of unknown (caught on A3's mean and
max), a stream fed every event regardless of when it was recorded (the late
500.00 reappears in the 09:00 row), and an encoder that turns an unknown into
a zero.

Integers are compared exactly. Floats use `math.isclose(rel_tol=1e-9)`, because
the two paths add the same numbers in a different order and the sums are equal
to a few ulps rather than bit for bit. Calling that agreement is a choice, and
a feature where one part in a billion changes a decision is not one this design
can defend.

## Adding a feature to a store that is already full

```python
backfill_online(extended_view, new_spec, store, events, as_of)
```

It computes the one new feature with the serving path's own accumulator, then
writes one field per existing row. A backfilled value computed by the batch
path would be a value the parity check never looked at.

Every row ends in one of five outcomes, counted and printed:

| Outcome | When |
|---|---|
| written | the row is there and stamped at the same instant |
| absent | there is no row, so none is created |
| as_of_mismatch | the row was computed at a different instant |
| expired_mid_write | the key went away between the read and the write |
| no_history | the stream has no events for this entity |

`absent` is the interesting one. `HSET` on a key that does not exist creates it
**with no ttl at all**, so a backfill that touched an expired account would
leave an immortal row holding exactly one feature, never refreshed and never
expired. The store reads `_as_of` first and leaves an absent row absent.

Those two round trips cannot be made atomic without a script, so the key can
expire in between. The ttl is read back after the write and a key that comes
out persistent is deleted, which turns a rare lost row into a miss instead of
into one that never expires.
`tests/test_backfill.py::test_a_row_that_expires_mid_write_is_removed_rather_than_left_immortal`
drives it with a client that deletes the key on every `hset`.

## What it does not do

- **A join per spine row.** The batch path merges the spine against the events
  on the entity and masks per feature, which materialises every (spine row,
  event) pair for an entity. Fine for the tens of thousands of rows this is
  written for, wrong at a hundred million, where this is a window function in
  the warehouse and not pandas at all.
- **Keep an entity forever.** The accumulator remembers an entry for every
  entity it has seen, so an account that transacts once and never returns stays
  in memory. Bounded by entity count and not by traffic, and the fix is an
  entity-level expiry that is not written.
- **Recompute history.** The online backfill fills the current row at the
  current `as_of`. There is no replay that rewrites a week of vectors, because
  the online store holds one point in time per entity by design, and the
  training frame is where history lives.
- **Any streaming runtime.** `accumulator_as_of` replays a dataframe row by row,
  which is the slowest possible feed and exists so the same code is exercised
  from a test and from the command line. In production that loop is a consumer.
- **Five aggregations only.** count, sum, mean, max, distinct count. No
  percentiles, no ratios between features, no time since last event. Each would
  need its own answer to what an empty window means, and three of them would
  need state the accumulator does not keep.
- **Approximate distinct counts.** `distinct_countries_7d` holds a real set. At
  cardinalities where that stops fitting the answer is HyperLogLog, and then
  the batch path and the serving path have to agree on an approximation, which
  is a harder parity argument than this repo makes.

## Where skew can still get in

- **An entity the stream has never seen.** The batch path fills its empty
  windows with zeros; serving can only answer MISS. The parity check reports
  this rather than hiding it, and the spine for a training run should be built
  from entities the stream has seen.
- **A writer whose `as_of` lags its own clock.** A job computing an hour-old
  window writes rows that are new in the store and old in event time, and they
  read FRESH. One row cannot tell a lagging job apart from a slow clock, and
  refusing to serve either would make a slow clock an outage, so the read path
  judges only how long the store has held the row. Watching the lag belongs
  where `as_of` and the store clock are both in hand, which is the write.
- **A late event that arrives after the window it belongs to has closed.** The
  batch path, run again later, will include it and change a historical feature
  value. Serving will not. Reproducibility of a training frame is a property of
  the event log here, not of this code.
- **Float order.** See the tolerance above. It is a decision, not an absence of
  one.

## Tests

```bash
pytest -q        # 152 tests, no Redis, no network, no service
```

`fakeredis` stands in for the serving store and the batch path runs on frames
built in the test, so everything above is reachable offline. The files worth
reading first are `tests/test_parity.py`, for the mutations it requires the
check to catch, and `tests/test_offline_point_in_time.py`, for the late arrival
and the window edge.

`txn_count_24h` and `distinct_countries_7d` are named to match two of the
inputs in the request schema of
[model-serving](https://github.com/Sam-Farooq/model-serving), which scores
transactions and computes none of its own features.
