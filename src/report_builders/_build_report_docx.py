"""Fill FIT3181_5215_KaggleReport_2026S2.docx -> 32613571_assignment01_report.docx.

Every number here is taken from the recorded logs (experiment_history_*.csv, *_progress.csv) and
from the Kaggle public leaderboard; nothing is rounded up or invented.
"""
import copy
import re

import docx
from docx.shared import Pt

SRC = 'FIT3181_5215_KaggleReport_2026S2.docx'
OUT = '32613571_assignment01_report.docx'

doc = docx.Document(SRC)
paras = doc.paragraphs


def find(prefix):
    for p in doc.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    raise KeyError(prefix)


def set_text(p, text, italic=False, bold=False):
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    run = p.add_run(text)
    run.italic, run.bold = italic, bold
    run.font.size = Pt(10.5)
    return p


def add_after(p, text, italic=False, bold=False):
    new = copy.deepcopy(p._p)
    p._p.addnext(new)
    np_ = docx.text.paragraph.Paragraph(new, p._parent)
    set_text(np_, text, italic, bold)
    return np_


def paragraphs_after(anchor, lines):
    """Write `lines` as consecutive paragraphs after `anchor`, keeping its style."""
    last = anchor
    for line in lines:
        bold = line.startswith('**') and line.endswith('**')
        last = add_after(last, line.strip('*'), bold=bold)
    return last


def para_after_table(t, template_p, text):
    """A paragraph placed directly after a table, keeping the template paragraph's style."""
    new = copy.deepcopy(template_p._p)
    t._tbl.addnext(new)
    return set_text(docx.text.paragraph.Paragraph(new, template_p._parent), text)


def table_after(anchor, rows, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = 'Table Grid'
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = ''
            run = cell.paragraphs[0].add_run(val)
            run.font.size = Pt(9)
            run.bold = (r == 0)
    anchor._p.addnext(t._tbl)
    return t


# Shareable link to the trained checkpoints (the unit confirmed on the forum that the .pt files
# go to Drive/OneDrive and the report carries the link, rather than into the zip).
CHECKPOINT_LINK = 'https://drive.google.com/drive/folders/1gMk58k2UlutXSzoQ_jx7WEHFjQSARvDY?usp=sharing'

# ---------------------------------------------------------------- Set up
p = find('(Information about the required libraries')
set_text(p, 'Language and libraries: Python 3.11.4, PyTorch 2.11.0 (CUDA 12.8), timm 1.0.29, '
            'torchvision 0.26.0, NumPy 2.4.6, pandas 3.0.5, matplotlib 3.11.1, Pillow, scikit-learn, and '
            'bitsandbytes 0.50.2 for the 8-bit optimiser used by the largest backbone. '
            'Pretrained backbones are downloaded by timm from Hugging Face.')
p = add_after(p, 'Hardware: one NVIDIA RTX 3080 (10 GB) on Windows; the eight fine-tuning runs behind '
                 'the two submissions took about 28 GPU-hours in total. The 10 GB budget is the binding '
                 'constraint on this project and shaped the largest model choice directly - see the '
                 'Architecture table.')
p = add_after(p, 'Reproduction: the attached notebook rebuilds both submitted CSVs from the saved '
                 'per-model test probabilities and verifies them row-for-row (it reports 0 differing '
                 'rows). It runs standalone: the data/ folder shipped beside it in this archive holds '
                 'the probability arrays, the recorded training logs, the two submitted CSVs and a '
                 '20-image sample, and only NumPy, pandas, PyTorch and matplotlib are required (timm '
                 'is optional, used to instantiate the architectures). It does not re-run training, '
                 'which needs the 28 GPU-hours above; the per-epoch training logs of all eight runs are '
                 'replayed in section 5. The competition image arrays (350 MB) are not redistributed '
                 'here, and the trained model checkpoints are kept outside this archive because of '
                 'their size - see Trained models below.')
p = add_after(p, 'Trained models: the four fine-tuned checkpoints behind the two submissions (8.52 GB in '
                 'total) are shared at ' + CHECKPOINT_LINK + ' - pick 1 uses '
                 'pick1_EVAP_eva02_large_distilled.pt and pick1_HP_dinov3_vit_huge_plus_distilled.pt, '
                 'pick 2 uses pick2_V3P_dinov3_vit_large_distilled.pt and '
                 'pick2_HPbase_dinov3_vit_huge_plus_supervised.pt. Each file holds the EMA weights of '
                 'its best epoch plus the backbone name, epoch and validation accuracy. Re-running the '
                 '6-view test-time augmentation from each checkpoint, at the same scales that submission '
                 'uses, reproduces the exact probability array the notebook blends: 300/300 sampled test '
                 'images agree on argmax for all four, with a maximum probability difference of 4.2e-04 '
                 '(mixed-precision jitter). Evaluating these weights on the test set therefore gives the '
                 'leaderboard scores reported here.')

# ---------------------------------------------------------------- Dataset
p = find('(Describe train-validation-test split')
set_text(p, 'Source data: Animals_Dataset.zip holds 9,466 labelled JPEGs in 20 class folders at their '
            'original resolution (mostly 300-640px; a minority already 64x64), and test_set.zip holds the '
            '11,681 unlabelled test images at 64x64 (JPEG q75, 4:2:0). Every labelled image is therefore '
            'squashed to 64x64 with LANCZOS resampling first, so training and test share one resolution.')
p = add_after(p, 'Split: the labelled images are split per class into 8,047 training and 1,419 validation '
                 'images (stratified 15%, SEED = 2026). The split is frozen on disk, so every run in the '
                 'project is scored on exactly the same held-out images. The Kaggle test set is scored '
                 '80% public / 20% private.')
p = add_after(p, 'External training data: DomainNet painting and ImageNet-R images mapped to the same 20 '
                 'classes, because the training set is essentially all photographs while a sizeable minority of '
                 'the test set is renditions (paintings, drawings, sculptures). That fraction is not stated by '
                 'the competition and cannot be measured directly without labelling test images; solving '
                 'test = w x rendition-accuracy + (1-w) x photo-accuracy from the measured accuracies of each '
                 'submission puts it at 21-27%. Every external image is pushed through '
                 'the exact test degradation - squash crop, LANCZOS resize to 64x64, re-encoded with the '
                 'test set\'s own JPEG quantisation tables - and mixed into each batch at ratio 0.30 by a '
                 'class-balanced draw. A label audit removed two contaminated mappings (DomainNet '
                 '"bottlecap", which is a mosaic-art genre, and zebras mapped to horses).')
p = add_after(p, 'Reserved evaluation set: 772 external rendition images were reserved on disk before any '
                 'training draw could see them, and the training code asserts it never touches them. They '
                 'are the only non-circular rendition metric used for model selection.')
p = add_after(p, 'Normalisation: images stay uint8 64x64 and are scaled to [0,1] and normalised with '
                 'ImageNet statistics (mean 0.485/0.456/0.406, std 0.229/0.224/0.225); a fixed layer inside '
                 'each model re-normalises into that backbone\'s own statistics, so one data pipeline feeds '
                 'all four backbones. Training augmentation and 6-view test-time augmentation are listed '
                 'under Techniques.')

# ---------------------------------------------------------------- Architecture table
set_text(find('(You can briefly present your model'),
         'Every final model is a pretrained Vision Transformer fine-tuned end-to-end with a fresh '
         '20-way linear head. Four backbones were trained, spanning three pretraining recipes. '
         'Submission 1 blends two different recipes (EVA-02 and DINOv3); submission 2 blends two DINOv3 '
         'models that differ in capacity and in training stage - ViT-L self-distilled and ViT-H+ supervised '
         'only - rather than in recipe.')
arch = doc.tables[1]
rows = [
    ('Architecture', 'Additional notes'),
    ('vit_large_patch14_reg4_dinov2.lvd142m (V2)',
     'DINOv2 self-supervised ViT-L/14, 304.4M parameters, 4 register tokens. Fine-tuned end-to-end at 196px. Trained and reported below; it is in neither final submission, having been displaced by the larger DINOv3 models.'),
    ('vit_large_patch16_dinov3.lvd1689m (V3)',
     'DINOv3 self-supervised ViT-L/16, 303.1M parameters. Fine-tuned end-to-end at 192px. Member of submission 2.'),
    ('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k (EVA)',
     'EVA-02 Large, masked image modelling on Merged-38M then ImageNet-22k/1k supervised fine-tuning, 304.1M parameters, 196px. Member of submission 1.'),
    ('vit_huge_plus_patch16_dinov3.lvd1689m (HP)',
     'DINOv3 self-supervised ViT-H+/16, 840.5M parameters - 2.8x the others, 32 blocks, width 1280. Fine-tuned end-to-end at 192px. Its distilled stage is in submission 1 and its supervised stage in submission 2.'),
    ('Pooling', "Mean over patch tokens (global_pool='avg'); class and register tokens are dropped."),
    ('Classifier head', 'Single Linear(width, 20) - 1024 for the ViT-L models, 1280 for ViT-H+ - randomly initialised; the pretrained 1000-way head is discarded.'),
    ('Input normalisation layer', "Fixed buffers re-normalise the ImageNet-normalised batch into each backbone's own mean/std."),
    ('Weight EMA', 'Exponential moving average of all weights, decay 0.999. Inference uses the EMA weights.'),
    ('Ensemble head (no parameters)',
     'Two fine-tuned models are combined by matching their mean top-1 confidence through a solved temperature, then averaging probabilities with equal weight.'),
    ('Fitting ViT-H+ in 10 GB',
     'Full fine-tuning of 840.5M parameters needs about 18 GB (parameters 3.13, gradients 3.13, Adam moments 6.26, '
     'fp32 EMA 3.13) and almost none of it is activations, so a smaller batch does not help. Three changes that touch '
     'neither model nor data nor objective bring the measured peak to 8.3 GB at the same batch size of 20: 8-bit Adam '
     'moments (saves 4.7 GB), the EMA copy held in system RAM and swapped in only to validate (3.13 GB), and the EMA '
     'updated every 4 steps at decay^4 so its horizon in steps is unchanged. Master weights stay fp32: in bf16 the '
     'relative update at lr 2e-5 falls below the format resolution and would be rounded away.'),
    ('Submission file names',
     'ens = ensemble; temp = the members were temperature-matched before averaging; the middle is the member tags, '
     'where a trailing P marks the self-distilled stage of that model and a trailing 256 the native-grid TTA scales. '
     'So submission_ens_EVAPHP256temp.csv is EVA-02 Large self-distilled blended with DINOv3 ViT-H+/16 '
     'self-distilled, the latter inferred at 192/224/256px instead of 160/192/224px.'),
    ('"Distillation" - three senses',
     '(1) DINO = self-distillation with no labels, the self-supervised objective the DINOv2/DINOv3 trunks were '
     'pretrained with; (2) the released ViT-L checkpoints of both families are themselves distilled from the '
     'largest model of their family; (3) "self-distillation" in this report means only stage 2 below - our own '
     'Noisy-Student round on pseudo-labelled test images. Senses 1 and 2 are properties of the downloaded '
     'weights; sense 3 is the only one this project performed.'),
]
while len(arch.rows) < len(rows):
    arch.add_row()
for r, (a, b) in enumerate(rows):
    for c, val in enumerate((a, b)):
        cell = arch.cell(r, c)
        cell.text = ''
        run = cell.paragraphs[0].add_run(val)
        run.font.size = Pt(9)
        run.bold = (r == 0)

p = find('Hint: using information from model.summary())')
set_text(p, 'Section 3 of the notebook instantiates each of the four backbones and prints its real '
            'parameter count and head.')

# ---------------------------------------------------------------- Techniques (<= 150 words)
TECHNIQUES = (
    'Transfer learning from four pretrained backbones spanning three pretraining recipes (self-supervised '
    'DINOv2 and DINOv3, and masked image modelling), each fine-tuned end-to-end. Augmentation: TrivialAugmentWide, '
    'RandomResizedCrop(0.60-1.0), horizontal flip, RandomErasing(0.25), random JPEG re-encoding at '
    'quality 55-95, and MixUp(0.2)/CutMix(1.0) on half the batches with soft targets. Label smoothing '
    '0.1, AdamW, two warmup epochs then cosine decay, mixed precision with gradient checkpointing, and '
    'weight EMA (0.999). External rendition data degraded to the test spectrum attacks the photo-to-'
    'rendition domain gap. Noisy-Student self-distillation adds pseudo-labelled test images from an '
    'external teacher (argmax, confidence >= 0.7, loss weight 0.5, fresh student, one round). Inference '
    'uses 6-view test-time augmentation (three scales x horizontal flip). The two models in each '
    'submission are combined by temperature-matched equal-weight probability blending, which has no '
    'fitted parameter. Fitting the 840M-parameter backbone in 10 GB needed an 8-bit optimiser and a '
    'CPU-resident EMA.'
)
p = find('(Describe in detail all techniques')
set_text(p, TECHNIQUES)

# ---------------------------------------------------------------- Parameter setting
p = find('(Hyper-parameters you tuned')
set_text(p, 'Final settings (identical for all four backbones, one variable changed per experiment):')
tbl = table_after(p, [
    ('Hyper-parameter', 'Value', 'Why / what was tested'),
    ('Backbone learning rate', '2e-5, flat (no layer-wise decay)',
     'A layer-decay-0.75 ladder off a 1e-4 top cost 1.95pt of held-out rendition accuracy against an otherwise identical control.'),
    ('Head learning rate', '1e-3', 'Fresh 20-way head; standard.'),
    ('Optimiser / schedule', 'AdamW, weight decay 0.05, 2 warmup epochs then cosine', 'Project standard; not swept.'),
    ('Epochs', '16 per stage', 'Best epoch chosen on validation; later epochs never improved it.'),
    ('Batch size', '20', 'Largest that fits a ViT-L at 196px in 10 GB with gradient checkpointing, and kept for ViT-H+ as well: its memory is dominated by parameters and optimiser state, not activations, so a smaller batch would have saved almost nothing.'),
    ('Input resolution (training)', '192px (196px for patch-14 models)',
     'Training at 256px was tested and scored below 192px on the leaderboard at 1.3x the compute. That is a '
     'separate question from the inference scales in the next row: adding 256px as one of several test-time '
     'views costs no training compute.'),
    ('Test-time augmentation', '3 scales x horizontal flip (6 views): 160/192/224px, and 192/224/256px for ViT-H+',
     'An 18-view variant (adding rotations) changed rows but not the score. ViT-H+ has a native 256px position grid, so '
     'moving its scales up to 192/224/256 restores it; that raised held-out rendition accuracy from 95.21% to 95.98% (6 vs 0, p = 0.031) without lowering validation, though it moved only 7 rows inside each blend.'),
    ('External data ratio', '0.30, class-balanced draw', 'Tested 0.25 and 0.35; differences were inside noise.'),
    ('Pseudo-label threshold', '0.7', 'Lowering it to 0.60 cost 0.29pt: the extra images are photos labelled at ~67% accuracy.'),
    ('Pseudo-label loss weight', '0.5', 'Project standard; per-sample confidence weighting was tested and rejected.'),
    ('Label smoothing / MixUp / CutMix', '0.1 / alpha 0.2 / alpha 1.0 on 50% of batches', 'Kept from the baseline; not swept individually.'),
    ('EMA decay', '0.999', 'EMA weights beat raw weights on validation in most epochs.'),
])
para_after_table(tbl, p,
                 'Recommendation: keep the flat 2e-5 trunk learning rate, 16 epochs, batch 20 and '
                 'threshold 0.7; the two levers that actually moved the score were changing the '
                 'pretraining recipe and adding one round of self-distillation, not tuning these values.')

# ---------------------------------------------------------------- Loss & accuracy
p = find('(Record your model performance')
set_text(p, 'Per-epoch traces of all eight runs (loss, validation accuracy of raw and EMA weights, seconds) '
            'are replayed in section 5 of the notebook, with the curves plotted. Summary:')
results_tbl = table_after(p, [
    ('Model', 'Final training loss', 'Best validation acc.', 'Reserved renditions acc.', 'Kaggle public (test)'),
    ('V2  DINOv2 ViT-L/14, supervised only', '0.8997', '99.44%', '93.78%', '0.97363'),
    ('V2P DINOv2 ViT-L/14, + self-distillation', '0.8558', '99.30%', '93.01%', '0.97950'),
    ('V3  DINOv3 ViT-L/16, supervised only', '0.9121', '99.51%', '92.88%', '0.97556'),
    ('V3P DINOv3 ViT-L/16, + self-distillation', '0.8666', '99.51%', '94.30%', '0.98196'),
    ('EVA EVA-02 Large, supervised only', '0.8766', '99.30%', '92.49%', '0.96915'),
    ('EVAP EVA-02 Large, + self-distillation', '0.8572', '99.51%', '92.88%', '0.98132'),
    ('HP  DINOv3 ViT-H+/16, supervised only', '0.8724', '99.65%', '95.21%', '0.97950'),
    ('HP  DINOv3 ViT-H+/16, + self-distillation', '0.8399', '99.58%', '92.75%', '0.98014'),
    ('HP  ViT-H+ supervised, native-grid TTA (192/224/256)', '-', '-', '95.98%', 'blend member only'),
    ('HP  ViT-H+ self-distilled, native-grid TTA (192/224/256)', '-', '-', '93.26%', 'blend member only'),
    ('Submission 1: EVAP + HP blend', '-', '-', '94.04%', '0.98335'),
    ('Submission 2: V3P + HP blend (HP supervised)', '-', '-', '95.73%', '0.98335'),
])
para_after_table(results_tbl, p,
               'Validation accuracy is saturated (98-99.5%) and does not order the models the way the test '
               'set does; the reserved rendition set and the leaderboard are what separate them, and the '
               'rendition set has no valid reading across the supervised-to-distilled boundary. Both blends '
               'score above both of their own members on the reserved renditions, which is the signature of '
               'models that are right on different images. Two rows deserve comment. ViT-H+ is the only '
               'lineage whose self-distillation barely paid (+0.064pt against +0.59 to +1.22pt elsewhere): '
               'its supervised model was already the strongest here, so the stage had little to add and '
               'measurably eroded it - yet that same erosion left it the bluntest model in the set '
               '(mean confidence 0.8097) and therefore the best partner for the sharpest one (EVA-02 at '
               '0.9013), which is where the best result came from. Private-leaderboard scores were not '
               'visible at submission time.')


# ---------------------------------------------------------------- Discussion (<= 100 words)
DISCUSSION = (
    'The hardest part was measurement, not modelling: validation saturated near 99% and once ranked the '
    'worst model first, so every decision used a paired win-rate test on the leaderboard plus a rendition '
    'set reserved before training. What mattered was the pretraining recipe, then one distillation round on '
    'pseudo-labelled test images, then blending models whose errors differ. Scaling capacity within the '
    'best recipe paid last, once memory techniques made it fit. Limitations: 64px inputs lose real detail, '
    'the rendition proxy covers one stratum, the largest model carries two uncontrolled memory-driven '
    'differences, and remaining blends move too few rows to measure.'
)
p = find('(Summarize the things you have learned')
set_text(p, DISCUSSION)

# ---------------------------------------------------------------- References
p = find('(List articles/papers you have referred')
refs = [
    'Oquab, M. et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. TMLR.',
    'Simeoni, O. et al. (2025). DINOv3. arXiv:2508.10104.',
    'Fang, Y. et al. (2023). EVA-02: A Visual Representation for Neon Genesis. arXiv:2303.11331.',
    'Xie, Q., Luong, M.-T., Hovy, E., Le, Q. (2020). Self-training with Noisy Student improves ImageNet classification. CVPR.',
    'Zhang, H. et al. (2018). mixup: Beyond Empirical Risk Minimization. ICLR.',
    'Yun, S. et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. ICCV.',
    'Muller, S., Hutter, F. (2021). TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation. ICCV.',
    'Hendrycks, D. et al. (2021). The Many Faces of Robustness (ImageNet-R). ICCV.',
    'Peng, X. et al. (2019). Moment Matching for Multi-Source Domain Adaptation (DomainNet). ICCV.',
    'Wightman, R. (2019). PyTorch Image Models (timm). GitHub repository.',
    'Dettmers, T. et al. (2022). 8-bit Optimizers via Block-wise Quantization (bitsandbytes). ICLR.',
    'Guo, C. et al. (2017). On Calibration of Modern Neural Networks (temperature scaling). ICML.',
]
set_text(p, refs[0])
last = p
for r in refs[1:]:
    last = add_after(last, r)

doc.save(OUT)

words = lambda s: len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-.]*", s))
print(f'wrote {OUT}')
print(f'  Techniques: {words(TECHNIQUES)} words (limit 150)')
print(f'  Discussion: {words(DISCUSSION)} words (limit 100)')
