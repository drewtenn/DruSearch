# DruSearch Roadmap

## Ranking Label Roadmap

The LTR trainer should keep moving obvious ecommerce relevance rules into
training labels and features, not serving-time ordering guardrails.

Priority label families:

1. Category and product-type intent: category path/token matches, plus accessory
   versus core-product mismatch.
2. Attribute intent: gender/age, size, color, material, style, and use case.
3. Commerce quality: availability, price competitiveness, reviews/ratings,
   return/defect risk, shipping speed, margin, and calibrated popularity.
4. Behavioral labels: click, add-to-cart, purchase, dwell, bounce, reformulation,
   and session-level purchase attribution.
5. Personalization labels: preferred brand/category/price band/size/color,
   repeat-purchase affinity, and suppression for non-repeat purchased items.
6. Negative labels: brand/category mismatch, accessory mismatch, adult/kids or
   gender mismatch, out-of-stock products, title stuffing, and low-quality items.

Completed:

- Strengthen category/product-type labeling so queries like `running shoes`
  learn category relevance before broader popularity or semantic similarity
  signals. Category and product-type pseudo labels are implemented in
  `pipelines.label.build_training_rows` with conservative category-only
  upgrades and core-product versus accessory compatibility checks.

- Add attribute-intent labels for gender/age, size, color, material, style,
  and use-case matches so attribute-heavy queries learn precise product
  relevance before popularity or semantic similarity signals dominate. Initial
  pseudo labels are implemented in `pipelines.label.build_training_rows` with
  single-attribute and multi-attribute label floors.

Current item:

- Add commerce-quality labels for availability, price competitiveness,
  reviews/ratings, return/defect risk, shipping speed, margin, and calibrated
  popularity.
