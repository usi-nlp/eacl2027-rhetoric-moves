# Discrimination study — Label Studio setup (annotator)

1. Install and start Label Studio (Python 3.9+):

       pip install label-studio
       label-studio

   It opens at http://localhost:8080 — create a local account (stays on your machine).

2. Set up the project (once):
   - Create Project → name it `Introduction Study`.
   - Labeling Setup → Custom template → paste the contents of `config.xml` → Save.
   - Settings → Instructions → paste the contents of `instructions.html`
     (tick "show before labeling"), and read it before starting.
   - Import → upload `tasks.json` (120 items).

3. Annotate: click **Label All Tasks** and work through the items in the order
   they appear — it resumes where you left off, so stop and continue whenever
   suits you. Submit each item once; please don't revisit or change earlier
   answers, and don't skip, filter, or reorder.

4. Every ~30 items (and when you finish): Export → **JSON** (the full one, not
   JSON-MIN) and send the downloaded file to the research team. Intermediate
   exports are just backups — export again over the same project any time.

Notes: judge each introduction on its own; please do NOT search the web for any
passage of the texts. Minor formatting glitches are processing artifacts and not
a signal either way.
