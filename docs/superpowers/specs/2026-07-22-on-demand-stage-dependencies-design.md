# On-Demand Stage Dependencies

## Goal

Keep every enabled pipeline stage's Preview and Run actions available and let a
downstream action calculate missing upstream data with the parameters currently
shown in the UI. Existing upstream results are reused silently when their
computational parameters match. When they do not match, the user chooses whether
to recalculate, reuse the existing result, or cancel.

The pipeline dependency order is:

`raw images -> displacement -> force -> stress`

Stress also requires a mask. Raw images and masks are source inputs and cannot be
derived by the pipeline.

## User-Facing Behavior

Preview and Run remain enabled for every non-disabled stage while an experiment
is selected. The action validates source inputs when clicked instead of using
disabled buttons to communicate missing data. An optional stage that the user has
explicitly disabled remains disabled.

When an action needs an upstream artifact:

- A missing artifact is calculated automatically with the current UI parameters.
- An existing artifact with matching computational parameters is reused without a
  prompt.
- An existing artifact with different computational parameters opens a modal with
  **Recalculate with current parameters**, **Use existing data**, and **Cancel**.
- Missing raw images or a required stress mask produces a clear warning and no
  calculation starts.

Preview resolves dependencies only for the viewer's current frame. Run resolves
full-stack dependencies. Thus Force Preview calculates a missing current-frame
displacement before force, while Stress Run calculates missing full displacement
and force stacks before stress.

Frame-only prerequisite results are transient inputs to the active preview chain.
They do not replace a full-stack artifact already held by the data manager. Full
Run prerequisites become the current in-memory artifacts and use the data
manager's normal downstream invalidation rules.

## Architecture

Add a central interactive dependency coordinator owned by the workflow shell.
Stage header actions ask it to execute a target stage and mode (`preview` or
`run`). The coordinator builds the shortest dependency chain, resolves each
artifact in order, and starts the originally requested operation when its inputs
are ready.

The stage controllers remain responsible for stage-specific validation,
parameter retrieval, computation, cancellation, and visualization. They expose
small coordinator-facing entry points that can return a current-frame result or
produce a full-stack result. Controllers do not call one another.

The coordinator owns:

- dependency ordering;
- missing/stale/reusable decisions;
- the stale-result prompt;
- asynchronous continuation from one stage to the next;
- stopping the chain after cancellation or failure; and
- operation-level progress messages.

This avoids coupling Stress directly to Force or Force directly to Displacement,
and it keeps interactive behavior separate from the persisted batch pipeline.

## Parameter Freshness

Freshness compares the parameters stored on an artifact with the current
parameters that affect that stage's numerical result. Display-only controls do
not make an artifact stale.

The excluded visualization parameters are:

- Displacement: maximum display magnitude, vector stride, and arrow scale.
- Force: maximum display magnitude, vector stride, and arrow scale.
- Stress: maximum display stress.

Comparison is performed on normalized parameter values so dataclasses, enums,
NumPy scalars, and equivalent primitive values compare consistently. If an old or
externally loaded artifact lacks enough parameter metadata to establish a match,
it is treated as stale and the user is asked.

Choosing **Use existing data** applies only to the current requested chain. It
does not rewrite the artifact's parameter metadata or suppress future prompts.
Choosing **Recalculate** uses the current UI parameters and replaces a full-stack
artifact only after that prerequisite completes successfully.

## Execution and Cancellation

Only one interactive dependency chain may own a target stage at a time. While it
is active, the target stage shows its existing running/cancel state. Progress is
reported through the global status line, including the requested operation and
the prerequisite currently being calculated, for example: “Force preview:
calculating missing displacement.”

Cancel is forwarded to the currently active prerequisite or target controller.
After cancellation, the coordinator discards any pending continuation, so no
downstream calculation starts. A prerequisite error likewise terminates the chain
and reports both the requested action and the prerequisite that failed.

The stale-result prompt is evaluated independently for each required upstream
stage. For example, a Stress action may reuse current displacement but prompt for
stale force. Cancel at any prompt aborts the complete request.

## Persistence

This feature does not change the distinction between interactive computation and
batch persistence. Interactive previews remain unpersisted. Full prerequisite
runs update in-memory artifacts through the existing data manager; persistence
continues to follow the workflow's existing interactive/batch rules.

On-disk artifacts may still be loaded as candidates. After loading, they undergo
the same freshness comparison rather than being assumed current merely because
they exist.

## Tests

Tests will be written before production changes and cover:

- Preview and Run enabled with an active experiment despite missing derived data;
- disabled optional stages remaining disabled;
- Force Preview calculating only current-frame displacement when missing;
- Force Run calculating full-stack displacement when missing;
- Stress Preview resolving displacement and force in order for the current frame;
- Stress Run resolving missing full-stack prerequisites in order;
- matching results being reused without a prompt;
- stale and metadata-poor results offering recalculate, reuse, and cancel;
- display-only parameter changes not marking results stale;
- recalculation using current UI parameters;
- transient preview prerequisites not overwriting full-stack artifacts;
- cancellation preventing downstream continuation;
- prerequisite failures preventing the target calculation and identifying the
  failed stage; and
- missing source images or stress masks producing a warning after the click.

Focused controller/coordinator tests will use synchronous fake workers for
deterministic chaining. Existing stage lifecycle, workflow shell, preview worker,
data ownership, and reload-on-selection suites will guard integration behavior.
