import torch
from diffusers import ZImagePipeline

model_path = "/data/disk1/model/qwen/Z-Image-Turbo"

pipe = ZImagePipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    trust_remote_code=True,
)
pipe.to("cuda:6")

# prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

prompt = """
一张采用双色调背景、具有现代科技感的商业广告海报，上部为浅蓝色，下部为深灰色渐变，展示了三款不同的科技产品。在上半部分，用深蓝色大号无衬线字体写着“智能&高效”。其下方是较小的白色文字“(前沿科技解决方案)”。主标题下方印有深蓝色无衬线字体的英文“Tech-Vision Pro”。左侧，一台银色笔记本电脑放在透明亚克力底座上，屏幕显示着数据可视化界面。右侧，另一台放在更高底座上的平板电脑展示着3D建模界面。右下方，第三个产品是智能手表，放在黑色充电座上，表盘显示着健康监测数据。在产品之间，散落着一些微型芯片和电路板元素。左下角有两个全息投影的立方体。海报为每个产品都附有描述性文字：“极速版¥5999”，配置为“i7处理器·16GB内存”，规格为“规格: 13.3英寸”；“专业版 ¥7999”，配置为“i9处理器·32GB内存”，规格为“规格: 15.6英寸”；以及“至尊版 ¥9999”，配置为“顶级显卡·64GB内存”，规格为“规格: 17.3英寸”。品牌名“科技视界”位于左下角。右下角写着“NEW RELEASE”和“(限量预售)”。该图像是一张高质量的影棚摄影作品，采用冷色调LED灯光，营造出专业而未来感的美学效果。风格是科技摄影风格，比例是3:4。
"""
# 2. Generate Image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("example1.png")