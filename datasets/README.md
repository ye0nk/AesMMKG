# Dataset layout

The local book and movie data are intentionally excluded from Git. Preserve the
original IDs, splits, graph triples, and feature vectors when reproducing the
experiments. Add the verified dataset download location and preparation
instructions before publishing a complete reproduction package.

```text
datasets/
  book/
    train.txt
    test.txt
    valid.txt
    kg_final.txt
    image_embeddings.txt
    entity_list.txt
    relation_list.txt
    item_list.txt
    user_list.txt
  movie/
    ... same filenames ...
```

## Files consumed by training

| File | Format |
| --- | --- |
| `train.txt` | No header. Each line contains a user ID followed by space-separated training item IDs. |
| `test.txt` | Same format, containing held-out item IDs. |
| `kg_final.txt` | No header. Each line contains integer `head relation tail` IDs separated by whitespace. |
| `image_embeddings.txt` | One header line, then tab-separated `org_id`, `remap_id`, and a space-separated floating-point vector. |

`remap_id` in the embedding file indexes an entity row. The loader fills missing
entity feature rows with zeros. All loaded feature vectors must have the same
dimension. Users are appended after entity rows in the model's shared table.

`valid.txt` is present locally but is not consumed by the current training entry
points. Do not describe these scripts as using validation-based model selection.

## Mapping files and preprocessing

`entity_list.txt` and `relation_list.txt` map original identifiers to integer IDs.
The image relation preprocessing script expects a header and tab-separated
`org_id` and `remap_id` columns. It expects three tab-separated columns in
`item_list.txt`: `org_id`, `remap_id`, and `amazonKG_id`. `user_list.txt` records
the user mapping and is retained with the original data.

The image preprocessing script reads its own configured inputs under `my_kg/`;
it does not automatically use `datasets/`. Image entities use the `img:` prefix,
and image files are located using the script's existing label-cleaning rule and
the `.jpg` extension.

Both local relation mappings define `hasImage` as ID 20. The training loader adds
2 to KG relation IDs, making `--has_image_rel_id 22` the matching setting. Preserve
the mappings together with the graph and embeddings when distributing the data.

## Distribution

Publish the preparation code and this format guide in GitHub. Distribute the
original data, embeddings, and checkpoints as a separate versioned artifact when
redistribution is permitted, and link it from the root README. Include checksums,
split definitions, and exact preprocessing settings so that readers can identify
the matching files. The current snapshot does not provide a verified public
download URL.
