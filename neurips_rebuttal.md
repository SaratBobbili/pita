# NeurIPS Rebuttal — Submission 27150

## 1 Reviewer sSE7

Thank you very much for your review and constructive comments. Here we would like to address the reviewer's concerns and hope that this helps improve your evaluation of our paper.

### Weakness #1:
The compared baselines are inadequate. Most notably, "best-of-N" is an important baseline for inference-time alignment methods that should be compared to. There are also other inference-time alignment methods, such as Controlled Decoding and Value-Augmented Sampling, that the authors did cite but only partially compared against.

**Our Response:**

<!-- TODO -->

### Weakness #2:
Even when PITA is shown to outperform Q#-HF, it is usually achieved at a much higher KL regime. When KL-matched (Table 2), which the authors agree is an important metric, PITA performs similarly to Q#-HF.

**Our Response:**

<!-- TODO -->

### Weakness #3:
Related to the previous point, I am concerned about the robustness of the method (since higher KL is more susceptible to reward hacking). Only the AlpacaFarm experiments in the paper use non-verifiable human labels, and PITA's improvement is only modest.

**Our Response:**

<!-- TODO -->

### Weakness #4:
PITA requires token-level logit access to the base model, making it not applicable to "opaque models whose internal weights are inaccessible," which is the stated motivation of the paper in the introduction.

**Our Response:**

<!-- TODO -->

### Weakness #5:
The bolding in Table 1 is a bit unconventional.

**Our Response:**

<!-- TODO -->

## 2 Reviewer 27QZ

Thank you very much for your review and constructive comments. Here we would like to address the reviewer's concerns and hope that this helps improve your evaluation of our paper.

### Weakness #1:
The practical contribution and the distinction from reward/value-guided decoding are not sharp enough. Although the paper emphasizes avoiding an explicit reward model, PITA still learns a preference predictor / Q-like guidance model that plays a reward-like role in decoding. Under the Bradley-Terry formulation used in the paper, this is closer to replacing an explicit final-output reward model with an action-value / preference guidance model than to eliminating preference modeling altogether.

**Our Response:**

<!-- TODO -->

### Weakness #2:
The experimental comparison is mainly against inference-time value-guidance baselines (especially Q# / Q#-HF), rather than against strong post-training baselines such as DPO or PPO/RLHF. Since the paper frames PITA as an alignment method, the lack of matched-data / matched-compute comparison to standard preference-optimization methods weakens the empirical claim.

**Our Response:**

<!-- TODO -->

### Weakness #3:
It is unclear whether a relatively small guidance model can reliably reweight the token distribution of a much larger LLM, especially when the desired behavior requires substantial distributional shift or capabilities not already present in the base model. Calibration error in the learned guidance function could oversteer decoding and degrade generation quality as the KL / guidance parameter varies.

**Our Response:**

<!-- TODO -->

### Weakness #4:
Empirical gains are mixed and not always clearly significant. On GSM8K, PITA improves over the reference but remains below the Q# baseline; on MATH-500, gains are small and maj@8 slightly decreases relative to the reference. The paper needs a more careful discussion of practical significance, statistical confidence intervals or multiple-seed results, and clearer compute comparisons against ordinary DPO/PPO, best-of-N / reranking, and value-guided decoding at matched inference budgets.

**Our Response:**

<!-- TODO -->

### Question #1:
In what precise sense is PITA not learning a reward model, given that it learns a preference/Q-like guidance function used to reweight token probabilities?

**Our Response:**

<!-- TODO -->

### Question #2:
How does PITA compare to DPO and PPO/RLHF trained on the same preference data and evaluated under matched compute?

**Our Response:**

<!-- TODO -->

### Question #3:
How sensitive is the method to calibration errors in the learned guidance model and to the KL/guidance parameter eta?

**Our Response:**

<!-- TODO -->

### Question #4:
What is the inference-time overhead relative to standard decoding, best-of-N, and value-guided decoding baselines?

**Our Response:**

<!-- TODO -->

### Question #5:
Does PITA remain useful when the preferred behavior is not already likely under the reference model?

**Our Response:**

<!-- TODO -->

## 3 Reviewer FSAP

Thank you very much for your review and constructive comments. Here we would like to address the reviewer's concerns and hope that this helps improve your evaluation of our paper.

### Weakness #1:
I am less certain about the overall significance of the empirical contribution. The practical advantages of PITA over existing value-guided or preference-based alignment methods are not always fully clear.

**Our Response:**

<!-- TODO -->

### Weakness #2:
The experimental evaluation provides relatively limited evidence that PITA consistently outperforms strong alternatives across diverse settings, and some of the observed gains appear modest.

**Our Response:**

<!-- TODO -->

### Weakness #3:
Although the method removes the need for an explicit reward model, it still relies on preference feedback and iterative preference-guided estimation, so the overall reduction in alignment cost is less clear in practice.

**Our Response:**

<!-- TODO -->

### Weakness #4:
The distinction between PITA and existing reward-free or value-guided inference approaches needs to be articulated more sharply. Overall, I am not yet convinced that the empirical evidence and practical advantages establish a substantial advance over the existing literature on inference-time alignment.

**Our Response:**

<!-- TODO -->

### Question #1:
The paper argues that preference-guided value estimation can replace the traditional reward-modeling stage. Could the authors provide a more explicit comparison of the overall computational and data requirements between PITA and reward-model-based approaches? Clarifying the practical tradeoffs would help assess whether the proposed framework offers a meaningful reduction in alignment cost beyond removing an explicit reward model.

**Our Response:**

<!-- TODO -->

### Question #2:
While the theoretical development is a major strength of the paper, the empirical evaluation appears relatively limited compared to the breadth of the paper's claims. Could the authors provide additional analysis on the consistency of PITA across tasks and preference distributions, particularly in settings where the observed gains are small or mixed? This would help clarify the robustness of the approach.

**Our Response:**

<!-- TODO -->

### Question #3:
The paper positions PITA as distinct from reward-model-based and value-guided inference methods. Could the authors more clearly articulate the conceptual and practical differences between PITA and existing inference-time alignment approaches such as value-guided decoding, Q-guided inference, or other reward-free preference optimization methods? A sharper comparison would strengthen my assessment of the paper's novelty.

**Our Response:**

<!-- TODO -->

### Question #4:
PITA relies on iterative preference-guided estimation during inference. How sensitive is the method to the quality and quantity of preference feedback, and what are the primary failure modes when preference signals are noisy or inconsistent? Additional analysis would help clarify the practical applicability of the framework.

**Our Response:**

<!-- TODO -->
