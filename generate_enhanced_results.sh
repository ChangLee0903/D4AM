chime_root=../CHiME4/data/audio/16kHz/isolated_1ch_track
python main.py --task write --test_root $chime_root --method NOIS --test_set chime --output_dir $chime_root
for method in INIT CLSO SRPR GCLB D4AM; do
    python main.py --task write --test_root $chime_root --ckpt ckpt/$method.pth --output_dir results/${method}_chime_result --bsz 8 --method $method --test_set chime
done
aurora_root=../Aurora4
python main.py --task write --test_root $aurora_root --method NOIS --test_set aurora --output_dir $aurora_root
for method in INIT CLSO SRPR GCLB D4AM; do
    python main.py --task write --test_root $aurora_root --ckpt ckpt/$method.pth --output_dir results/${method}_aurora_result --bsz 8 --method $method --test_set aurora
done

