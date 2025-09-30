import pathlib
import torch
from concurrent.futures import ThreadPoolExecutor
from threading import local
from tqdm import tqdm
from sacrebleu import corpus_bleu
from sonar.inference_pipelines.text import TextToTextModelPipeline


class SONAREvaluator:
    def __init__(self, dataset_path="datasets/flores200/devtest"):
        self.dataset_path = pathlib.Path(dataset_path)
        self.num_gpus = torch.cuda.device_count()
        # self.num_gpus = 
        self.thread_local = local()
        
        # Get supported languages once
        self._get_supported_languages()
        
    def _get_supported_languages(self):
        """Get supported languages from tokenizer."""
        temp_pipeline = self._create_pipeline(0)
        self.supported_langs = temp_pipeline.tokenizer._langs
        del temp_pipeline
        
    def _create_pipeline(self, gpu_id):
        """Create pipeline for specific GPU."""
        device = torch.device(f"cuda:{gpu_id}" if self.num_gpus > 0 else "cpu")
        return TextToTextModelPipeline(
            encoder="text_sonar_basic_encoder",
            decoder="text_sonar_basic_decoder", 
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=torch.float16
        )
    
    def _get_pipeline(self, gpu_id):
        """Get thread-local pipeline instance."""
        if not hasattr(self.thread_local, 'pipeline'):
            self.thread_local.pipeline = self._create_pipeline(gpu_id)
        return self.thread_local.pipeline
    
    def _evaluate_pair(self, args):
        """Evaluate single language pair."""
        (src_lang, tgt_lang), (src_path, tgt_path), gpu_id = args
        
        pipeline = self._get_pipeline(gpu_id).to(torch.float16)
        

        src_text = src_path.read_text().splitlines()
        translation = pipeline.predict(src_text, source_lang=src_lang, target_lang=tgt_lang, batch_size=1024, beam_size=1)
        ref_text = tgt_path.read_text().splitlines()
        bleu = corpus_bleu(translation, [ref_text], tokenize="flores200", force=True)
        return f"{src_lang}-{tgt_lang}", bleu.score

    
    def _prepare_pairs(self):
        """Prepare all language pairs."""
        pairs = []
        eng_ref = self.dataset_path / "eng_Latn.devtest"
        
        for lang_file in self.dataset_path.glob("*.devtest"):
            lang = lang_file.name.split(".")[0]
            if lang in self.supported_langs:
                if lang == "eng_Latn":
                    pairs.append((("eng_Latn", "eng_Latn"), (eng_ref, eng_ref)))
                    continue
                pairs.extend([
                    ((lang, "eng_Latn"), (lang_file, eng_ref)),
                    (("eng_Latn", lang), (eng_ref, lang_file)),
                    ((lang, lang), (lang_file, lang_file))
                ])
                
        return pairs
    
    def evaluate(self):
        """Run evaluation with threading."""
        pairs = self._prepare_pairs()
        print(f"Evaluating {len(pairs)} language pairs on {max(1, self.num_gpus)} GPUs")
        
        # Assign GPU to each task
        tasks = [(pair, gpu_id % max(1, self.num_gpus)) for gpu_id, pair in enumerate(pairs)]
        eval_args = [(*task[0], task[1]) for task in tasks]
        
        # Run evaluation
        scores = {}
        with ThreadPoolExecutor(max_workers=max(1, self.num_gpus)) as executor:
            results = list(tqdm(executor.map(self._evaluate_pair, eval_args), 
                              total=len(eval_args), desc="Evaluating"))
        
        for lang_pair, score in results:
            scores[lang_pair] = score
        
        return self._calculate_averages(scores)
    
    def _calculate_averages(self, scores):
        """Calculate average scores."""
        x_to_eng = [s for k, s in scores.items() if k.endswith("-eng_Latn")]
        eng_to_x = [s for k, s in scores.items() if k.startswith("eng_Latn-") and not k.endswith("-eng_Latn")]
        x_to_x = [s for k, s in scores.items() if k.split('-')[0] == k.split('-')[1]]
        
        results = {
            'all_scores': scores,
            'overall': sum(scores.values()) / len(scores),
            'x_to_eng': sum(x_to_eng) / len(x_to_eng) if x_to_eng else 0,
            'eng_to_x': sum(eng_to_x) / len(eng_to_x) if eng_to_x else 0,
            'x_to_x': sum(x_to_x) / len(x_to_x) if x_to_x else 0
        }
        
        # Print results
        print(f"\nResults:")
        print(f"Overall: {results['overall']:.2f}")
        print(f"X→eng: {results['x_to_eng']:.2f}")
        print(f"eng→X: {results['eng_to_x']:.2f}") 
        print(f"X→X: {results['x_to_x']:.2f}")
        
        return results


# Usage
if __name__ == "__main__":
    evaluator = SONAREvaluator()
    results = evaluator.evaluate()