# Competition Data

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

The frozen 50,000-product competition catalog is not included in this public
repository or in the runtime-assets release.

Obtain the catalog through the official TechJam participant-kit distribution
instructions and place the decompressed file at:

`data/catalog.jsonl`

Expected row count: 50,000.