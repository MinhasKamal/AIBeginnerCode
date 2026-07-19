from huggingface_hub import list_repo_files
from huggingface_hub import HfApi

def get_model_subfolders(model_id: str):
    print(f"Fetching files for {model_id}...")
    try:
        # Get a list of every file path inside the repository
        all_files = list_repo_files(repo_id=model_id)
        
        # Extract the top-level directory names
        subfolders = set()
        for file_path in all_files:
            if "/" in file_path:
                top_folder = file_path.split("/")[0]
                subfolders.add(top_folder)
                
        return sorted(list(subfolders))
    except Exception as e:
        print(f"Error accessing repository: {e}")
        return []

def get_model_download_size(model_id: str):
    api = HfApi()
    total_gb = 0
    try:
        repo_info = api.model_info(repo_id=model_id, files_metadata=True)
        total_bytes = sum(sibling.size for sibling in repo_info.siblings if sibling.size)
        total_gb = total_bytes / (1024 ** 3)
        files = list((sibling.rfilename, sibling.size) for sibling in repo_info.siblings if sibling.size > 1024 ** 2)
        print(f"Files: {files}")
    except Exception as e:
        print(f"Error fetching data: {e}")
    
    return total_gb

# --- Example Usage ---
if __name__ == "__main__":
    model_id_list = [
        "prs-eth/marigold-v1-0",
        "prs-eth/marigold-depth-v1-1"
        "stabilityai/sdxl-vae",
        "stabilityai/stable-diffusion-xl-base-0.9",
        "stabilityai/stable-diffusion-xl-refiner-0.9",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-xl-1.0-tensorrt",
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-large-tensorrt",
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1-tensorrt",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.2-dev",
    ]

    for model_id in model_id_list:
        print(f"## {model_id} subfolders: {get_model_subfolders(model_id)}")
        print(f"size: {get_model_download_size(model_id):.2f} GB \n\n")





# Fetching files for prs-eth/marigold-v1-0...
# ## prs-eth/marigold-v1-0 subfolders: ['scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae']
# Files: [('text_encoder/model.fp16.safetensors', 680820392), ('text_encoder/model.safetensors', 1361597016), ('text_encoder/pytorch_model.bin', 1361679905), ('text_encoder/pytorch_model.fp16.bin', 680899947), ('tokenizer/vocab.json', 1059962), ('unet/diffusion_pytorch_model.bin', 3463980773), ('unet/diffusion_pytorch_model.fp16.bin', 1732130133), ('unet/diffusion_pytorch_model.fp16.safetensors', 1731927776), ('unet/diffusion_pytorch_model.safetensors', 3463772592), ('vae/diffusion_pytorch_model.bin', 334715313), ('vae/diffusion_pytorch_model.fp16.bin', 167405651), ('vae/diffusion_pytorch_model.fp16.safetensors', 167335342), ('vae/diffusion_pytorch_model.safetensors', 334643276)]
# size: 14.42 GB 


# Fetching files for stabilityai/sdxl-vae...
# ## stabilityai/sdxl-vae subfolders: []
# Files: [('diffusion_pytorch_model.bin', 334712113), ('diffusion_pytorch_model.safetensors', 334643268), ('sdxl_vae.safetensors', 334641164)]
# size: 0.94 GB 


# Fetching files for stabilityai/stable-diffusion-xl-base-0.9...
# ## stabilityai/stable-diffusion-xl-base-0.9 subfolders: ['scheduler', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'unet', 'vae']
# Files: [('01.png', 4126912), ('sd_xl_base_0.9.safetensors', 13875726784), ('text_encoder/model.fp16.safetensors', 246144864), ('text_encoder/model.safetensors', 492265880), ('text_encoder/pytorch_model.bin', 492307041), ('text_encoder/pytorch_model.fp16.bin', 246187076), ('text_encoder_2/model.fp16.safetensors', 1389382880), ('text_encoder_2/model.safetensors', 2778702976), ('text_encoder_2/pytorch_model.bin', 2778810597), ('text_encoder_2/pytorch_model.fp16.bin', 1389493517), ('tokenizer/vocab.json', 1059962), ('tokenizer_2/vocab.json', 1059962), ('unet/diffusion_pytorch_model.bin', 10270603837), ('unet/diffusion_pytorch_model.fp16.bin', 5135676955), ('unet/diffusion_pytorch_model.fp16.safetensors', 5135149760), ('unet/diffusion_pytorch_model.safetensors', 10270077736), ('vae/diffusion_pytorch_model.bin', 334712113), ('vae/diffusion_pytorch_model.fp16.bin', 167405651), ('vae/diffusion_pytorch_model.fp16.safetensors', 167335342), ('vae/diffusion_pytorch_model.safetensors', 334643268)]
# size: 51.70 GB 


# Fetching files for stabilityai/stable-diffusion-xl-refiner-0.9...
# ## stabilityai/stable-diffusion-xl-refiner-0.9 subfolders: ['scheduler', 'text_encoder_2', 'tokenizer_2', 'unet', 'vae']
# Files: [('01.png', 4126912), ('sd_xl_refiner_0.9.safetensors', 6075948232), ('text_encoder_2/model.fp16.safetensors', 1389382880), ('text_encoder_2/model.safetensors', 2778702976), ('text_encoder_2/pytorch_model.bin', 2778810597), ('text_encoder_2/pytorch_model.fp16.bin', 1389493517), ('tokenizer_2/vocab.json', 1059962), ('unet/diffusion_pytorch_model.bin', 9038639429), ('unet/diffusion_pytorch_model.fp16.bin', 4519587067), ('unet/diffusion_pytorch_model.fp16.safetensors', 4519210760), ('unet/diffusion_pytorch_model.safetensors', 9038264392), ('vae/diffusion_pytorch_model.bin', 334712113), ('vae/diffusion_pytorch_model.fp16.bin', 167405651), ('vae/diffusion_pytorch_model.fp16.safetensors', 167335342), ('vae/diffusion_pytorch_model.safetensors', 334643268)]
# size: 39.62 GB 


# Fetching files for stabilityai/stable-diffusion-xl-base-1.0...
# ## stabilityai/stable-diffusion-xl-base-1.0 subfolders: ['scheduler', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'unet', 'vae', 'vae_1_0', 'vae_decoder', 'vae_encoder']
# Files: [('01.png', 4608613), ('sd_xl_base_1.0.safetensors', 6938078334), ('sd_xl_base_1.0_0.9vae.safetensors', 6938078334), ('sd_xl_offset_example-lora_1.0.safetensors', 49553604), ('text_encoder/flax_model.msgpack', 492248682), ('text_encoder/model.fp16.safetensors', 246144152), ('text_encoder/model.onnx', 492587457), ('text_encoder/model.safetensors', 492265168), ('text_encoder/openvino_model.bin', 492242672), ('text_encoder/openvino_model.xml', 1057789), ('text_encoder_2/flax_model.msgpack', 2778657095), ('text_encoder_2/model.fp16.safetensors', 1389382176), ('text_encoder_2/model.onnx_data', 2778639360), ('text_encoder_2/model.safetensors', 2778702264), ('text_encoder_2/openvino_model.bin', 2778640120), ('text_encoder_2/openvino_model.xml', 2790191), ('tokenizer/vocab.json', 1059962), ('tokenizer_2/vocab.json', 1059962), ('unet/diffusion_flax_model.msgpack', 10269915611), ('unet/diffusion_pytorch_model.fp16.safetensors', 5135149760), ('unet/diffusion_pytorch_model.safetensors', 10270077736), ('unet/model.onnx', 7293842), ('unet/model.onnx_data', 10269854720), ('unet/openvino_model.bin', 10269856428), ('unet/openvino_model.xml', 22577438), ('vae/diffusion_flax_model.msgpack', 334623853), ('vae/diffusion_pytorch_model.fp16.safetensors', 167335342), ('vae/diffusion_pytorch_model.safetensors', 334643268), ('vae_1_0/diffusion_pytorch_model.fp16.safetensors', 167335342), ('vae_1_0/diffusion_pytorch_model.safetensors', 334643268), ('vae_decoder/model.onnx', 198093688), ('vae_decoder/openvino_model.bin', 197961232), ('vae_encoder/model.onnx', 136775724), ('vae_encoder/openvino_model.bin', 136655184)]
# size: 71.63 GB 


# Fetching files for stabilityai/stable-diffusion-xl-1.0-tensorrt...
# ## stabilityai/stable-diffusion-xl-1.0-tensorrt subfolders: ['lcm', 'lcmlora', 'sdxl-1.0-base', 'sdxl-1.0-refiner']
# Files: [('lcm/clip.opt/model.onnx', 322531134), ('lcm/clip2.opt/model.onnx', 1517189726), ('lcm/unetxl.opt/dbf91c42-985c-11ee-9041-0242ac110002', 5136090880), ('lcm/unetxl.opt/model.onnx', 3369087), ('lcm/vae.opt/model.onnx', 99186612), ('lcmlora/clip.opt/model.onnx', 322531134), ('lcmlora/clip2.opt/model.onnx', 1517189726), ('lcmlora/unetxl-8c8ce9e8b00b259425e5f3eaa4b1d705-1.00.opt/1376e228-9608-11ee-9b07-0242ac110002', 5136090880), ('lcmlora/unetxl-8c8ce9e8b00b259425e5f3eaa4b1d705-1.00.opt/model.onnx', 3369087), ('lcmlora/vae.opt/model.onnx', 99186612), ('sdxl-1.0-base/clip.opt/model.onnx', 322531134), ('sdxl-1.0-base/clip2.opt/model.onnx', 1517189726), ('sdxl-1.0-base/unetxl.opt/435d4c0a-2d32-11ee-8476-0242c0a80101', 5136090880), ('sdxl-1.0-base/unetxl.opt/model.onnx', 6136637), ('sdxl-1.0-refiner/clip2.opt/model.onnx', 1517189726), ('sdxl-1.0-refiner/unetxl.opt/6e186582-2d74-11ee-8aa7-0242c0a80102', 4519958016), ('sdxl-1.0-refiner/unetxl.opt/6ed855ee-2d70-11ee-af8e-0242c0a80101', 847120896), ('sdxl-1.0-refiner/unetxl.opt/model.onnx', 4040948)]
# size: 26.10 GB 


# Fetching files for runwayml/stable-diffusion-v1-5...
# ## runwayml/stable-diffusion-v1-5 subfolders: ['feature_extractor', 'safety_checker', 'scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae']
# Files: [('safety_checker/model.fp16.safetensors', 608018440), ('safety_checker/model.safetensors', 1215981830), ('safety_checker/pytorch_model.bin', 1216061799), ('safety_checker/pytorch_model.fp16.bin', 608103564), ('text_encoder/model.fp16.safetensors', 246144864), ('text_encoder/model.safetensors', 492265874), ('text_encoder/pytorch_model.bin', 492305335), ('text_encoder/pytorch_model.fp16.bin', 246187076), ('tokenizer/vocab.json', 1059962), ('unet/diffusion_pytorch_model.bin', 3438354725), ('unet/diffusion_pytorch_model.fp16.bin', 1719327893), ('unet/diffusion_pytorch_model.fp16.safetensors', 1719125304), ('unet/diffusion_pytorch_model.non_ema.bin', 3438366373), ('unet/diffusion_pytorch_model.non_ema.safetensors', 3438167536), ('unet/diffusion_pytorch_model.safetensors', 3438167540), ('v1-5-pruned-emaonly.ckpt', 4265380512), ('v1-5-pruned-emaonly.safetensors', 4265146304), ('v1-5-pruned.ckpt', 7703807346), ('v1-5-pruned.safetensors', 7703324286), ('vae/diffusion_pytorch_model.bin', 334707217), ('vae/diffusion_pytorch_model.fp16.bin', 167405651), ('vae/diffusion_pytorch_model.fp16.safetensors', 167335342), ('vae/diffusion_pytorch_model.safetensors', 334643276)]
# size: 44.01 GB 


# Fetching files for stabilityai/stable-diffusion-3.5-large...
# ## stabilityai/stable-diffusion-3.5-large subfolders: ['scheduler', 'text_encoder', 'text_encoder_2', 'text_encoder_3', 'text_encoders', 'tokenizer', 'tokenizer_2', 'tokenizer_3', 'transformer', 'vae']
# Files: [('sd3.5_large.safetensors', 16460379262), ('sd3.5_large_demo.png', 18080198), ('text_encoder/model.fp16.safetensors', 247323896), ('text_encoder/model.safetensors', 247323896), ('text_encoder_2/model.fp16.safetensors', 1389382176), ('text_encoder_2/model.safetensors', 1389382176), ('text_encoder_3/model-00001-of-00002.safetensors', 4994582104), ('text_encoder_3/model-00002-of-00002.safetensors', 4530066248), ('text_encoder_3/model.fp16-00001-of-00002.safetensors', 4994582104), ('text_encoder_3/model.fp16-00002-of-00002.safetensors', 4530066248), ('text_encoders/clip_g.safetensors', 1389382176), ('text_encoders/clip_l.safetensors', 246144152), ('text_encoders/t5xxl_fp16.safetensors', 9787841024), ('text_encoders/t5xxl_fp8_e4m3fn.safetensors', 4893934904), ('tokenizer/vocab.json', 1059962), ('tokenizer_2/vocab.json', 1059962), ('tokenizer_3/tokenizer.json', 2424035), ('transformer/diffusion_pytorch_model-00001-of-00002.safetensors', 9985185992), ('transformer/diffusion_pytorch_model-00002-of-00002.safetensors', 6307519304), ('vae/diffusion_pytorch_model.safetensors', 167666902)]
# size: 66.67 GB 


# Fetching files for stabilityai/stable-diffusion-3.5-large-tensorrt...
# ## stabilityai/stable-diffusion-3.5-large-tensorrt subfolders: ['ONNX']
# Files: [('ONNX/clip_g/model_optimized.onnx', 1390314479), ('ONNX/clip_l/model_optimized.onnx', 247672441), ('ONNX/t5/14f104de-6e4b-11f0-bc34-79ad8d380ae8', 9524621312), ('ONNX/transformer/bf16/fe820521-6e4b-11f0-805c-79ad8d380ae8', 16720058368), ('ONNX/transformer/bf16/model_optimized.onnx', 2829659), ('ONNX/transformer/fp8/model.onnx_data', 16292541952), ('ONNX/transformer/fp8/model_optimized.onnx', 5930622), ('ONNX/vae/model_optimized.onnx', 99235681), ('ONNX/vae_encoder/model_optimized.onnx', 68668591)]
# size: 41.31 GB 


# Fetching files for stabilityai/stable-video-diffusion-img2vid-xt-1-1-tensorrt...
# ## stabilityai/stable-video-diffusion-img2vid-xt-1-1-tensorrt subfolders: ['unet-temp.opt']
# Files: [('svd11.webp', 6544938), ('unet-temp.opt/a75198aa-dcc4-11ee-a242-0242c0a80101', 3049975040), ('unet-temp.opt/model.onnx', 3124930)]
# size: 2.85 GB 


# Fetching files for black-forest-labs/FLUX.1-dev...
# ## black-forest-labs/FLUX.1-dev subfolders: ['scheduler', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'transformer', 'vae']
# Files: [('ae.safetensors', 335304388), ('dev_grid.jpg', 1301528), ('flux1-dev.safetensors', 23802932552), ('text_encoder/model.safetensors', 246144352), ('text_encoder_2/model-00001-of-00002.safetensors', 4994582224), ('text_encoder_2/model-00002-of-00002.safetensors', 4530066360), ('tokenizer/vocab.json', 1059962), ('tokenizer_2/tokenizer.json', 2424235), ('transformer/diffusion_pytorch_model-00001-of-00003.safetensors', 9983040304), ('transformer/diffusion_pytorch_model-00002-of-00003.safetensors', 9949328904), ('transformer/diffusion_pytorch_model-00003-of-00003.safetensors', 3870584832), ('vae/diffusion_pytorch_model.safetensors', 167666902)]
# size: 53.91 GB 


# Fetching files for black-forest-labs/FLUX.2-dev...
# ## black-forest-labs/FLUX.2-dev subfolders: ['scheduler', 'text_encoder', 'tokenizer', 'transformer', 'vae']
# Files: [('ae.safetensors', 336211292), ('flux2-dev.safetensors', 64446596128), ('teaser_editing.png', 11257468), ('teaser_generation.png', 23264407), ('text_encoder/model-00001-of-00010.safetensors', 4883550696), ('text_encoder/model-00002-of-00010.safetensors', 4781593336), ('text_encoder/model-00003-of-00010.safetensors', 4886472224), ('text_encoder/model-00004-of-00010.safetensors', 4781593376), ('text_encoder/model-00005-of-00010.safetensors', 4781593368), ('text_encoder/model-00006-of-00010.safetensors', 4886472248), ('text_encoder/model-00007-of-00010.safetensors', 4781593376), ('text_encoder/model-00008-of-00010.safetensors', 4781593368), ('text_encoder/model-00009-of-00010.safetensors', 4886472248), ('text_encoder/model-00010-of-00010.safetensors', 4571866320), ('tokenizer/tokenizer.json', 17078037), ('transformer/diffusion_pytorch_model-00001-of-00007.safetensors', 9935797200), ('transformer/diffusion_pytorch_model-00002-of-00007.safetensors', 9890181048), ('transformer/diffusion_pytorch_model-00003-of-00007.safetensors', 9814681480), ('transformer/diffusion_pytorch_model-00004-of-00007.safetensors', 9814681536), ('transformer/diffusion_pytorch_model-00005-of-00007.safetensors', 9814681536), ('transformer/diffusion_pytorch_model-00006-of-00007.safetensors', 9814681536), ('transformer/diffusion_pytorch_model-00007-of-00007.safetensors', 5361898792), ('vae/diffusion_pytorch_model.safetensors', 336213556)]
# size: 165.44 GB 
