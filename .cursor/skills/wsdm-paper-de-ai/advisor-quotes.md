# Advisor wording to keep

Quoted from `paper_recsys/0821zhidao.txt` and the 2026-08-22 oral note. Do not paraphrase these into a second hedge.

## Consistency (0823)

改稿先对齐已有数据、假说和结论。不要为了去AI味道或缩短段落，推翻前序数字、假设方向或已经写进引言/贡献/RQ/结论的判断。

## Do not auto-polish (0821)

把这些改完我会建议同学就直接提交，不要再让大模型自动大范围润色全文了。现在最容易出问题的反而就是它把已经核准的数据、协议或措辞又偷偷改掉。最后一轮最好只人工逐句diff。

## AI-tone diagnosis (2026-08-22)

目前这篇文章和RecSys版相比，感觉AI写作痕迹重了不少。主要是很多"结论 -> 限定 -> 再降调”的模板，这种我审稿的时候看多了AI写的论文，感觉就很明显。本质就是经过LLM按审稿意见统一润色后的风格，AI会写出极其工整的“主动防止overclaim”结构而且连续出现多次。

## Claim / protocol order (0821)

修改顺序上，我建议先改protocol/claim，把Early time interval写清，items temporally disjoint改成interactions，RQ1/RQ2那几段机制解释统一降成hypothesis/association。再改图表里的winner语气，Figure1最好明确这是point-estimate ordering，而且差距落在1std内的地方用\approx号，例如Toys Last更适合写TF+MV \approx TF>TF+LLM>ID，LOO三个文本模型很多地方也更接近TF+LLM \approx TF+MV≈TF。论文现在恰恰强调winner不robust，图里就不要画得像确定性冠军。

Early uses the first 60% for training, the next 10% for validation, and the next 10% for testing; the final 20% is unused.

the evaluation window is shifted earlier while validation/test horizon lengths remain fixed.

training and evaluation interactions are separated by a global temporal cutoff.

## RQ3 locus (0821)

redundancy is already present in the raw Qwen outputs before SVD/whitening under our prompt design

不是 redundancy originates in the Qwen encoder itself。

the observed geometry is consistent with insufficient view complementarity

exactly predicted by the redundancy analysis 改成 consistent with the redundancy analysis。

## RQ2 sentence (2026-08-22, use as written)

TF-IDF exhibits the smallest Main-to-Early drop among the three text variants; we treat this as an empirical robustness observation rather than evidence for a causal mechanism.

(0821 draft opened with *Despite catalog shift*. 0823 dropped that prefix: Tables 5–6 are Main valid→test context, not the Main→Early experiment.)

## Citations as an AI tell (0813)

今年我审NeurIPS的时候，PC特别提醒了AI写作的文章一个明显的问题是引用文献会出错误。

一般来说都推荐同学Google Scholar直接复制一下BibTex。

## Citation claim error (0820)

Introduction 里 “LLM-generated item descriptions improve accuracy over TF-IDF or Word2Vec [TALLRec, UniSRec]” 不是那两篇的贡献。改为 pretrained/large language representations can improve text-enhanced recommendation and transferability。

## Process language (0820)

A matched-protocol hybrid retrain was paused… 这种“实验暂停了”的开发过程不要留在论文里。Appendix 里 initially evaluated without…、earlier catalog-fit run 也建议全部去掉。论文只呈现最终公平协议。
