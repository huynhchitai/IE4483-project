# (g) Applying the classifier to unlabelled hotel reviews

## Expected performance

Our classifiers are trained on labelled product reviews concerning clothing
and footwear. Although training is supervised, **sentiment labels are not
required at inference time**. The fitted models can therefore generate
predictions for hotel reviews without rating scores, assuming that the trained
source models remain available. However, the absence of hotel labels prevents
ordinary supervised fine-tuning directly on the target data and prevents
measurement of target-domain accuracy or F1.

| Operation | Are genuine hotel labels required? | Implication |
|---|---|---|
| Predict with a source-trained classifier | No | The existing model can assign sentiment labels to hotel reviews. |
| Train or fine-tune directly with a supervised hotel classification loss | Yes | Hotel text alone does not provide the required target labels. |
| Adapt using pseudo-labels | No, but source supervision is reused | Model-generated labels provide an imperfect training signal. |
| Measure hotel accuracy or F1 | Yes | Predictions must be compared with independent ground truth. |

The models may recognize general sentiment expressions such as “excellent”,
“terrible” and “very disappointing”. Nevertheless, their performance may
deteriorate because of **domain shift**: hotel reviews differ from product
reviews in vocabulary, subject matter and the relationship between particular
expressions and sentiment. As a general illustration, “small” may describe a
desirable compact device but an undesirable hotel room; this example is not
an observation from our clothing-and-footwear dataset. Our exploratory analysis
also identifies source-specific terms related to fit, materials and returns,
whose usefulness may change in the hotel domain.

For the TF-IDF baseline, terms outside the fitted vocabulary contribute no
features. A GRU may similarly struggle with unseen words or unfamiliar uses of
known words. Consequently, neither model's performance on product reviews
should be assumed to carry over unchanged to hotel reviews. **Without genuine
hotel sentiment labels, accuracy and F1 cannot be calculated to verify the
quality of these predictions.**

### Expected effects of the domain change

| Source of shift | TF-IDF + Logistic Regression | Embedding sequence + GRU |
|---|---|---|
| Hotel terms absent from the source vocabulary | Unseen terms contribute no features. | Unseen terms may map to an unknown token, depending on the tokenizer. |
| Familiar words used in a different context | Source-trained coefficients may reflect inappropriate sentiment associations. | Source-trained embeddings and sequence patterns may not capture the new usage. |
| Shared sentiment expressions | Retained words and bigrams may provide transferable signals. | Learned lexical and sequential cues may transfer partially. |
| Different sentiment prevalence | Source-derived class weights and the decision threshold may be unsuitable. | Source-derived loss weights and the decision threshold may also be unsuitable. |

These are anticipated risks and transfer opportunities, not measured hotel-domain
results. Neither representation guarantees robustness to a change of domain.

## Proposed modifications

### Comparison of adaptation options

| Approach | Applicable model | Additional information required | Potential benefit | Main limitation |
|---|---|---|---|---|
| Pretrained word embeddings | GRU | External pretrained vectors and labelled source reviews | Broader lexical coverage and a better initialization | Coverage and hotel-specific sentiment understanding are not guaranteed. |
| Pseudo-labelling and self-training | Both pipelines | Unlabelled hotel text, a source-trained model and retained source supervision | Incorporates target-domain examples into training | Confident mistakes and class imbalance can be reinforced. |
| Manual annotation and fine-tuning | Both pipelines | Newly annotated hotel reviews | Enables supervised target adaptation and independent evaluation | Requires annotation effort and relaxes the no-target-label condition. |

### Implementation considerations

1. **Use pretrained word embeddings for the GRU.** Replace randomly initialized
   embeddings with pretrained word embeddings to incorporate broader lexical
   knowledge, then train the sentiment classifier on the labelled product
   reviews. To benefit from hotel terms absent from the source vocabulary, the
   implementation must also support those tokens and their pretrained vectors;
   replacing embeddings only for existing source tokens does not resolve every
   out-of-vocabulary case. This approach may support transfer without requiring
   hotel labels, but it does not guarantee correct interpretation of
   hotel-specific contexts. It applies to the GRU representation rather than
   directly to the existing TF-IDF features.

2. **Apply pseudo-labelling and self-training.** Use a source-trained model to
   predict hotel review sentiment, select high-confidence predictions as
   provisional labels, and continue training with these examples alongside
   genuinely labelled product reviews. Monitor the selection process so that
   it does not include only positive predictions or assume that the hotel class
   distribution matches the product dataset. Pseudo-labelled examples can be
   assigned lower training weights, and the number of self-training iterations
   can be limited. Crucially, **confident predictions can still be incorrect**,
   especially under domain shift, so self-training may reinforce existing errors
   rather than improve performance. Pseudo-labels must not be treated as ground
   truth for evaluation.

3. **If manual annotation is permitted, label a small hotel dataset.** Use a
   portion of these annotations to fine-tune the classifier and, where needed,
   a separate validation portion to select settings. Retain an independent,
   representative labelled evaluation set that is not used for training,
   pseudo-labelling or model selection. This allows a direct comparison between
   the source-only and adapted models using accuracy, macro-F1 and per-class
   recall. Manual annotation relaxes the original no-target-label condition and
   should therefore be presented as an optional extension.

If all hotel reviews must remain unlabelled, the first two approaches remain
possible, provided that labelled source data or the trained source models can
be reused. Their benefits should be described as **expected improvements to
be investigated**, not demonstrated increases in accuracy. Reliable claims of
improved hotel sentiment performance require evaluation against independent
human-verified labels.

*Status: these are proposed adaptations; no hotel-domain experiment has been
performed. Source-domain evidence is documented in
[data_analysis.md](data_analysis.md).*
