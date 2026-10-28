# legacy

Earlier MinerU-based scripts from the HDP exercise QA/VQA run. They are kept for reference and are not
part of the package or its tests. The maintained MinerU route is `parse-ocr mineru`.

| Script | Purpose |
|---|---|
| `hdp_pdf_vqa_pipeline.py` | MinerU + DataFlow PDF_VQA prompt runner (needs `DATAFLOW_ROOT`) |
| `extract_math_environments.py` | select numbered environments from cached MinerU blocks |
| `correct_hdp_questions_visual.py` | crop-based visual correction of exercise statements |
| `correct_math_environments_visual.py` | visual transcription/verification of environments |
| `proofread_hdp_corrected.py` | math-frozen spelling repair |
| `build_sharegpt_output.py` | export questions as ShareGPT records |

Endpoints come from `VLM_CHAT_URL`; keys from `DF_API_KEY`.
