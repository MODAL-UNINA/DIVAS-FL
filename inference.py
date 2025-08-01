# %%
from utils.inference_utils import *
import pandas as pd

# %% 
if __name__ == "__main__":
    hps, args = get_hparams()  # True for inference mode
    if args.flag:
        vocoder = "hifigan"  # hifigan or bigvgan
        print("Running inference...")
        
        ckpt_model = f'models/{args.target}.pth'
        output_dir = f"results/generated_{args.target}"
        src_path = "run_txt/server/test_wav.txt"
        
        config2 = {
            "src_path": src_path,
            "ckpt_model": ckpt_model,    
            "voc": "bigvgan" if hps.vocoder.voc == "bigvgan" else "hifigan",
            "ckpt_voc": "./vocoder/voc_bigvgan.pth" if hps.vocoder.voc == "bigvgan" else "./vocoder/voc_hifigan.pth",
            "output_dir": output_dir,
        }

        hps2 = get_hparams_from_dict2(config2)
        print('>> Initializing Generation Process...')
        generate(hps2, hps, args)
    else:
        # === CONFIG ===
        real_folder = f"./results/generated_{args.target}/real_audio"
        generated_folder = f"./results/generated_{args.target}/generated_audio"

        results = evaluate_all(real_folder, generated_folder)
        spk_real = get_spk_embs(real_folder)
        spk_gen = get_spk_embs(generated_folder)
        
        # Evaluate EER
        eer_score = compute_eer(spk_real, spk_gen)
        print(f"\n🎙️ Equal Error Rate (EER): {eer_score:.4f}")

        # Global stats
        print("\n=== Evaluation summary ===")
        all_metrics = ['clap_cosine','wav_cosine','spk_dist','stoi', 'wer']

        df = pd.DataFrame.from_dict(results, orient='index')
        for metric in all_metrics:
            if metric in df.columns:
                print(f"{metric:12s}: mean = {df[metric].mean():.4f}")

        # Saving results
        df_summary = pd.DataFrame({
            'metric': df.columns.tolist() + ['eer_score'],
            'mean': [df[col].mean() for col in df.columns] + [eer_score],
        })
        df_summary.to_csv(f'results/generated_{args.target}/evaluation.csv', index=False)

# %%
