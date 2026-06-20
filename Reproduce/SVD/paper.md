A Training-Free Style-Personalization via SVD-Based Feature Decomposition
Kyoungmin Lee[] , Jihun Park[] , Jongmin Gim[*] , Wonhyeok Choi, Kyumin Hwang, Jaeyeul Kim and Sunghoon Im[†] DGIST, Daegu, Republic of Korea
{ kyoungmin, pjh2857, jongmin4422, smu06117, kyumin, jykim94, sunghoonim } @dgist.ac.kr
==> picture [410 x 165] intentionally omitted <==
----- Start of picture text -----<br>
Style Reference “A kettle” “An armchair” “A table” “A pair of shoes” “A car” “A ball”<br>— d 4 ws e<br>Style Reference “A pen” “A coral” “A bell” “A helmet” “A bird” “A book”<br>dspace a... "Waaey a eee esis Manaeeics<br>net a ¥ bf ' ne ne ‘ Bu k= gs » 4 wot<br>Style Reference “An hourglass” “A telescope” “A a castle with tall “A piano” “A horse running  “A train”<br>spires reflected in water” in a field”<br>----- End of picture text -----<br>

Figure 1. Style-personalized image generation results produced by our method. Given reference style images and text prompts, our method generates images with consistent style and diverse content.
Abstract
We present a training-free framework for style-personalized image generation that operates during inference using a scale-wise autoregressive model. Our method generates a stylized image guided by a single reference style while preserving semantic consistency and mitigating content leakage. Through a detailed step-wise analysis of the generation process, we identify a pivotal step where the dominant singular values of the internal feature encode stylerelated components. Building upon this insight, we introduce two lightweight control modules: Principal Feature Blending, which enables precise modulation of style
> 1Equal contribution. 
> 2Corresponding author. 
through SVD-based feature reconstruction, and Structural Attention Correction, which stabilizes structural consistency by leveraging content-guided attention correction across fine stages. Without any additional training, extensive experiments demonstrate that our method achieves competitive style fidelity and prompt fidelity compared to fine-tuned baselines, while offering faster inference and greater deployment flexibility.
1. Introduction
Text-to-Image (T2I) models [6, 16, 38, 41, 42, 45, 50] have rapidly transformed the creative landscape, enabling artists, designers, and casual users alike to generate high-quality visuals from natural language prompts. Fueled by massive
1
image-text datasets [4, 7, 30, 46], these models now support an unprecedented range of content diversity and stylistic expression. As creative tools become increasingly democratized, users are seeking more than just visually plausible outputs—they desire personalized generation that reflects specific visual identities [27, 29, 43, 44, 58] or preferred artistic styles [18, 31, 44, 49, 65]. These emerging demands call for generation systems that are not only high-quality but also customizable, efficient, and responsive to individual preferences.
Existing solutions [1, 13, 31, 47] have made progress in this direction, often relying on fine-tuning mechanisms to encode style-specific characteristics. However, such methods typically involve training a new model instance per style, posing scalability challenges in real-world applications. Additionally, most systems are built on diffusionbased T2I architectures [26, 38, 45]. Although these model produce high-quality results, their iterative denoising process leads to slow inference, which makes them less suitable for real-time or interactive applications.
Motivated by the limitations, we propose a novel stylepersonalized image generation framework that combines efficiency, flexibility, and stylistic fidelity. Our method generates high-quality images guided by a single reference style image during inference without any additional training. To achieve this, we leverage a large-scale text-to-image model, specifically a scale-wise autoregressive model [16], which offers significantly faster inference compared to diffusion models while maintaining strong visual fidelity.
To better understand and maximize the capabilities of this scale-wise autoregressive model, we conduct a detailed analysis of its generation process. Our analysis reveals that a specific step in the generation process plays a crucial role in determining both content and style. At this step, the dominant singular values of the feature play a key role, as they effectively capture and separate the style-related components. Building on this insight, we develop a Principal Feature Blending mechanism that enables precise control over style, leveraging a specific step feature and allowing the model to faithfully reflect the stylistic traits inherent in the style features of a reference image. Additionally, we introduce a Structural Attention Correction strategy, which stabilizes the generation process by leveraging content-related information to preserve structural consistency. By integrating these components, our training-free framework achieves high-quality image generation, as illustrated in Fig. 1. It also achieves competitive performance in both quantitative and qualitative evaluations, while maintaining significantly faster inference time.
In summary, our contributions include:
We present a training-free inference framework for stylepersonalized image generation from a single style reference, achieving competitive results with significantly
faster inference.
We conduct a detailed step-wise analysis of the scale-wise generation process and identify a key step that governs both content and style.
We observe that style-related components can be effectively extracted from the feature at this step through an SVD-based analysis of its dominant singular values.
We propose two lightweight control modules— Principal Feature Blending for precise style modulation and Structural Attention Correction for stabilizing structural coherence in generation.
2. Related Work
2.1. Neural Style Transfer
Image stylization, which alters the visual style of an image, has become an active area of research. A major breakthrough came with Neural Style Transfer (NST) [14], which used a pre-trained CNN (VGGNet) to separately extract content and style features. While effective, NST required costly per-image optimization. To tackle this issue, [20] proposed Adaptive Instance Normalization (AdaIN), aligning the mean and variance of the content features to those of the style features for faster style transfer. Building upon this, subsequent works such as [28, 33] introduced the Whitening and Coloring Transform (WCT), which aligns the full covariance structure of the features, resulting in more detailed and higher-quality stylization. With the rise of attention mechanisms in neural networks [11, 55], recent models further advanced stylization quality by utilizing attention to achieve remarkable stylization results [10, 15, 19, 32, 35, 62]. Parallel to this, vision-language models such as CLIP [40] have enabled text-driven style transfer [2, 24, 36], allowing intuitive control via natural language, without requiring explicit style images. This expands the scope of style transfer beyond traditional imagebased paradigms.
2.2. Text-to-Image Generation
Recent advances in large-scale image-text datasets [4, 7, 30, 46] have greatly enhanced the ability of models to bridge the gap between natural language and visual modalities, fueling progress in conditional image synthesis. This has spurred the development of large-scale Text-to-Image (T2I) generation frameworks—including diffusion-based models [26, 38, 41, 42, 45], GAN-based approaches [22], and visual autoregressive (AR) models [6, 16, 50]—which can generate diverse, high-quality images from natural language prompts. Diffusion-based models have become the dominant T2I paradigm due to their superior image quality and success in downstream tasks like style transfer and image editing [3, 9, 17, 21, 25, 39, 52, 61]. However, their high inference latency poses challenges for real-time applica-
2
tions. Meanwhile, visual AR models have evolved from traditional next-token prediction [12, 54] to more efficient masked token prediction [5, 6, 23]. The recent introduction of the next-scale prediction paradigm [51] has further accelerated inference without compromising output quality, establishing next-scale AR models [16, 50, 57] as promising alternatives to diffusion-based methods.
2.3. Personalized image generation
Recent advances in personalized image generation have led to methods that adapt novel visual concepts to user intent using pre-trained T2I models. These methods are generally categorized into content-oriented and style-oriented approaches. Content-oriented methods [27, 29, 43, 58] aim to capture object-specific or identity-preserving features from a small set of user-provided reference images. By finetuning pre-trained models or injecting learned embeddings, they generate images that maintain a high degree of fidelity to the target subject. Building on the technical foundations of these content-oriented personalization methods, recent work has extended similar principles to style-oriented personalized generation [1, 13, 18, 37, 44, 47, 49, 65]. In this paradigm, the objective shifts from preserving content identity to consistently controlling the visual style across diverse generations. Despite their effectiveness, these methods predominantly rely on diffusion-based models and often require fine-tuning, which introduces high computational costs and long inference times. In contrast, we propose a training-free, scale-wise autoregressive model that achieves fast style-personalized image generation based on a comprehensive analysis of the scale-wise autoregressive model.
3. Preliminary
Infinity Architecture. In our work, we leverage Infinity [16], a state-of-the-art T2I framework that employs the next-scale prediction paradigm introduced by [51] to generate high-fidelity, text-aligned images. During inference time, the Infinity architecture is composed of three key components: a pre-trained text encoder ET based on Flan-T5 [8], an autoregressive transformer M that performs scale-wise feature prediction, and a decoder D that reconstructs the final image from accumulated residual feature maps.
At each generation step s ∈ S , where S = { 1 , 2 , . . . , S} denotes the set of all generation steps, the autoregressive transformer M iteratively predicts a s -th scale quantized residual feature map Rs , conditioned on the input text prompt T and the previously generated feature Fs− 1. The process begins with initial features F 0 corresponding to the start-of-sequence ⟨ SOS ⟩ token. The prediction process is
defined as:
==> picture [233 x 38] intentionally omitted <==
where Qs , Ks , and Vs are the query, key, and value at s - th generation step projected from feature Fs respectively. Here, MSA ( · ) and MCA ( · ) denote the self-attention and cross-attention mechanisms within the transformer.
Each predicted residual Rs is upsampled to the resolution H × W using a bilinear upsampling function up H×W ( · ), and the resulting features are accumulated across scales to form the next-step input:
==> picture [207 x 28] intentionally omitted <==
where hs and ws denote the spatial dimensions of the residual features at step s , c denote the channel of the quantized feature. The final image I is produced by decoding the accumulated representation FS at the final generation step:
==> picture [144 x 11] intentionally omitted <==
4. Analysis of Scale-wise AR Model
(1) Step-wise analysis. We investigate the internal mechanisms of each step in the scale-wise autoregressive model’s generation process, focusing on its influence over two key visual attributes of the generated image: content and style representations. To facilitate this analysis, we construct two prompt pair sets ( T, T[ˆ] ) ∈ T[con] ∪ T[sty] from 100 base prompts, initially generated by ChatGPT:
Content pair set T[con] : This set contains 100 prompt pairs ( T[con] , T[ˆ][con] ) ∈ T[con] , each formed by randomly selecting two distinct object-centric base prompts (e.g., “A photo of a donut”, “A photo of a truck”).
Style pair set T[sty] : For each of the 100 base object prompts, a style pair ( T[sty] , T[ˆ][sty] ) ∈ T[sty] was created by assigning two different colors (from a set of 10 predefined colors) to the same object, while keeping object category fixed (e.g., “A photo of a red truck” and “A photo of a green truck”).
Then, for both T[con] and T[sty] , we generate modified images I[ˆ] , as shown in Fig. 2, by replacing the original text prompt T with an alternative prompt T[ˆ] at a specific generation step s ∈ S in Eq. 1, while keeping all other components unchanged. A substantial change in the resulting image in response to this step-specific prompt injection indicates that the corresponding step plays a critical role in shaping certain visual attributes.
To assess the impact of each generation step, we compute the CLIP similarity [40] between alternative prompt T[ˆ] and
3
==> picture [391 x 104] intentionally omitted <==
----- Start of picture text -----<br>
𝑠̂ = 1 2 3 4 5 6 7 8 9 10 11 12<br>3 7) 2 2 2 2) 2) 2) 2) 2<br>Ae )<br>𝑠̂ [!"] i step, text = “A photo of a white — teddybear” , else  “A photo of a S it  black teddybear” s<br>7 a,<br>F = tf 7) } sy J<br>: . a a a a a a a iy<br>7 𝑠̂ [!"] 7 step, text = 7 “A photo of a ——  donut” , else  “A photo of a  ; —. cupcake” + : Z<br>----- End of picture text -----<br>

Figure 2. Step-wise prompt injection analysis. We intervene at each generation step s ∈{ 1 , . . . , 12 } by replacing the prompt only at step s ˆ, while keeping all other steps fixed to the base prompt. Top : style prompt injection (“A photo of a black teddy bear” vs. “A photo of a white teddy bear”). Middle : content prompt injection (“A photo of a cupcake” vs. “A photo of a donut”). Bottom : CLIP similarity between the alternative prompt T[ˆ] and the corresponding image across steps.
==> picture [210 x 193] intentionally omitted <==
----- Start of picture text -----<br>
𝑇 <blue> “A photo of a <bunny> ” 1 𝐹! 1 SVD ⋯ 1 𝐹# Replace 1 ⋯ t 𝐹" 1 Baseline Full rep. guidedSVD-<br>𝑇 ["] <red> A “A photo of a <wizard> ” 𝐹 [#] ! 1 ⋯ ! ae 𝐹 [#] # ro ⋯ esa Cae 𝐹 [#] "<br>0.25 0.24 0.23 0.22 0.21 0.20 0.19 Style Similarity |  <blue>  → {𝐼 [#] , 𝐼 [#][!"#] , 𝐼 [#][$%&] }<br>Content Similarity<br>Baseline output 𝐼 ["]<br>| lia! —|<br>Full replacement<br>output 𝐼 ["][!"#]<br>Style  ↑<br>on 5 Content  ↑<br>+2 SVD-guided<br>’y output  Style  𝐼 [#][$%&] ↑<br>0.25 0.24 0.23 0.22 0.21 0.20 0.19<br>Content Similarity |  ee <bunny> → {𝐼 [#] eee , 𝐼 [#][!"#] , 𝐼 [#][$%&] }<br>----- End of picture text -----<br>

Figure 3. Key step feature analysis. Content and style similarity are measured for Baseline, Full replacement, and SVD-guided outputs using a set of prompt pairs T , with results averaged across all pairs.
its corresponding generated image as shown in the bottom ˆ row in Fig. 2. We observe that step s = 2 consistently produces the highest similarity with T[ˆ] across all 200 prompt pairs. This result suggests that step 2 plays a key role in shaping both content and style attributes. Consequently, the third feature F 3, generated after this step, plays a crucial role in determining the final output image.
(2) Key step feature analysis. As discussed in the previous section, the third feature F 3 plays a pivotal role in shaping both content and style. Building on this observation, and supported by prior studies [34, 37] showing that
early-step features in scale-wise autoregressive models often encode strong stylistic cues, we hypothesize that the principal components of F 3 are predominantly shaped by stylistic attributes.
To analysis this, we first construct a set of 100 prompt pairs T , where each pair ( T, T[ˆ] ) differs in both object category and color (e.g., “A photo of a red truck” and “A photo of a purple cat”). For each prompt T , we apply singular value decomposition (SVD) to the third feature, yielding F 3 = U Σ V[⊤] . We then construct a modified diagonal matrix Σ [′] , obtained by zeroing out all singular values except the largest one σ 1. Using this matrix, we reconstruct the dominant singular component as F 3[svd] = U Σ [′] V[⊤] . The corresponding residual is defined as F 3[res] = F 3 − F 3[svd][. We perform the same decomposition for] the feature F[ˆ] 3 obtained from the prompt T[ˆ] . For each prompt pair ( T, T[ˆ] ), we evaluate the effect of manipulating the dominant singular component by generating three outputs:
Baseline output I[ˆ] , generated using the original prompt T[ˆ] without any feature manipulation.
Full replacement output I[ˆ] [rep] , obtained by directly substituting the entire feature F[ˆ] 3 with F 3. ( F[ˆ] 3 ← F 3)
SVD-guided output I[ˆ] [svd] , obtained by replacing only the dominant singular component of F[ˆ] 3 with that of F 3, while preserving the residual component. ( F[ˆ] 3 ← F 3[svd] + F[ˆ] 3[res][)]
As shown in Fig. 3, the full-replacement output I[ˆ][rep] displays a substantial increase in CLIP similarity to both the object (e.g., “bunny”) and the color (e.g., “blue”) described in the substituted prompt T . In contrast, the SVDguided output I[ˆ][svd] shows a pronounced increase primarily in color-related CLIP similarity, while changes in objectrelated similarity remain much smaller. This result suggests that modifying only the dominant singular component primarily affects stylistic attributes, with minimal influence on
4
==> picture [397 x 156] intentionally omitted <==
----- Start of picture text -----<br>
uo 𝑅&"#$ co 𝑅'"#$ oo 𝑅("#$ ao 𝑅)"#$ ⋯ oY 𝑅%"#$ i < [Saieeieeemmemy] Principal Feature Blending (PFB)<br>Style Extractor Function Φ<br>𝐔𝚺𝐕 [(] 𝐔𝐖𝚺𝐕 [(]<br>Transformer<br>Structural Attention  𝐹!%&' 𝚺 exp 𝐖𝚺 Φ(𝐹!%&')<br>Correction (SAC)<br>i See Seer Sey =< TF _ ai<br>Φ<br>𝐹!"#$ 𝐹&"#$ 𝐹'"#$ 𝐹("#$ ⋯ 𝐹%&"#$ 𝐹%"#$<br>Lt Lt Lo L L 𝐹!"#$ aeeeseeeed Φ(𝐹!"#$) 𝐹!%&' −Φ(𝐹!%&') 𝐹 [%] !%&'<br>Text<br>Encoder Style reference Image — 𝐹("#$ Principal Feature  Decoder 𝑄)%&' SAC 𝐾)%&' 𝑉)%&'<br>“<content> in <style>” Encoder % Image  =Ld 𝐹(+,- Blending =. (PFB) j | =>) Element-wise AdditionElement-wise Subtraction Generation Path i Cj 𝐹)%&' :<br>𝑇<br>eZ ee _, Φ Style Extractor Function loveea  ceeas 𝐹 eee ) [+'] ee e Set<br>----- End of picture text -----<br>

Figure 4. Overall pipeline of our model . The text encoder processes an identical text prompt T for both the content and generation paths, providing their embeddings to the autoregressive transformer. At stage s = 3, Principal Feature Blend is applied to extract the principal style representation from the reference style image and seamlessly integrate it into the features of the generation path. Starting from s = 3 (the fine stage), Structural Attention Correction aligns the generation path’s attention maps with those of the content path, ensuring stable and consistent structural guidance during refinement.
content. Together, these observations suggest that the first principal component of the third feature F 3 predominantly captures style-related characteristics, with limited contribution from content-related information.
5. Method
5.1. Overall pipeline
In this paper, we aim to generate a style-personalized image I[gen] by injecting a principal style feature into the final generation image, while preserving semantic consistency and suppressing content leakage.
As illustrated in Fig. 4, our method employs a dualstream generation architecture composed of a content path and a generation path , both conditioned on the same text prompt T (“<content> in <style>”). Using an identical prompt prevents semantic mismatch between the two streams and enables consistent structural communication during inference. The content path operates as the standard inference branch of the pre-trained model without any modification and follows the iterative update rule in (1), producing a sequence of content features {Fs[con] }[S] s =1[.] Its role is to provide structurally stable and semantically aligned guidance throughout the generation process. In parallel, the generation path follows the same update formulation but produces its own feature sequence {Fs[gen] }[S] s =1[,] which is subsequently modulated by our proposed styleblending mechanism. This path synthesizes the final stylized output while leveraging structural cues from the content path and incorporating style information in a controlled and targeted manner.
Building upon this dual-stream iterative process, we introduce two complementary modules— Principal Fea-
ture Blending (PFB) and Structural Attention Correction (SAC)—which operate exclusively on the generation path while leveraging cues from the other streams. PFB (Sec. 5.2) selectively injects principal style representations from the style reference image into the generation features, with a targeted intervention at the third step ( s = 3) to prevent style-unrelated feature leakage. Following this style injection, SAC (Sec. 5.3) is applied across the subsequent steps, where it incorporates content path signals to stabilize structural alignment and maintain semantic consistency throughout the refinement process.
5.2. Principal Feature Blending
Our process begins by extracting the style features Fs[sty] using a pretrained multi-scale image encoder EI from the baseline Infinity [16], as follows:
==> picture [186 x 13] intentionally omitted <==
Among these multi-scale features, the step-wise analysis in Sec. 4 identifies the third feature F 3[sty] as a pivotal representation that strongly influences both content and style during generation. Motivated by this observation, we focus on F 3[sty] as the primary carrier of style information and use it as the basis for our style modulation mechanism.
To effectively incorporate style information while suppressing irrelevant cues from the reference image, we introduce Principal Feature Blending , a mechanism that selectively transfers the principal components of the style feature into the generation process. This design is grounded in two analyses from Sec. 4: (1) the Step-wise analysis , which pinpoints F 3 as the most influential feature for stylistic control, and (2) the Key step feature analysis , which
5
reveals that the dominant singular values of F 3[sty] encode the core stylistic characteristics. Guided by these insights, our method extracts the principal style representation from F 3[sty] and blends it into the generation path with minimal disruption to the original content structure.
To achieve seamless blending in the feature space, we design a style extractor function Φ, which prioritizes the dominant contribution of the leading component while smoothly incorporating residual style representations. Based on the observation that the first singular value of F 3[sty] acts as the primary carrier of style information, we enhance its influence while retaining minor contributions from the remaining components to preserve stylistic continuity. Accordingly, Φ applies exponential reweighting to the singular values based on their spectral order, gradually reducing the impact of lower components:
==> picture [223 x 46] intentionally omitted <==
where r denotes the rank of the feature matrix, and α > 0 controls the exponential decay rate along the singular spectrum.
To substitute the generation path’s style representation with the one extracted from the style feature, we incorporate the refined style via Φ and update the generation feature:
==> picture [194 x 31] intentionally omitted <==
This formulation preserves the original structure information of the generation path while seamlessly injecting style information derived from the reference.
5.3. Structural Attention Correction
While PFB effectively integrates style cues into the generation path, we observed that its feature-level modulation can unintentionally disturb the structural coherence of generated results, sometimes causing spatial misalignment or shape distortion. To stabilize the generation process, we leverage the attention map of the content path as a structural prior, inspired by the self-attention mechanism in diffusionbased architectures [48, 53], where the interaction between Queries and Keys preserves spatial and structural relationships. Building on this, we introduce Structural Attention Correction (SAC) , which aligns the attention map of the generation path with that of the content path to ensure consistent structural guidance throughout the generation process.
SAC is applied to all subsequent steps following the application of Principal Feature Blending (PFB), denoted as S fine = { 3 , 4 , . . . , S} . These steps correspond to the stages
where content and style representations continue to interact. Formally, SAC injects the content queries and keys at each step s ∈ S fine as follows:
==> picture [195 x 27] intentionally omitted <==
Here, WQ and WK denote the linear projection matrices that transform input features into query and key representations in the self-attention layers. Q[con] s and Qs[gen] denote the content and generation queries at step s , and Ks[con] and Ks[gen] denote the corresponding keys.
6. Experiments
6.1. Implementation Details
We implement our method using a pre-trained Infinity 2B model [16] with all parameters frozen, which performs scale-wise prediction across 12 steps. The baseline employs a codebook of size 2[32] , with quantized feature maps of resolution 64 × 64 × 32. The exponential decay rate α for Principal Feature Blending is set to 1.0.
Based on our analysis, our method operates in a stepwise manner with targeted interventions across the generation process. At s = 3, we apply Principal Feature Blending , and at the fine stages ( S fine = { 3 , 4 , . . . , S} ), we apply Structural Attention Correction . Generating a 1024 × 1024 style-personalized image takes approximately 3.58 seconds on a single NVIDIA A6000 GPU.
48 GB VRAM
6.2. Evaluation Setup
Benchmark. We follow the evaluation protocol introduced in FineStyle [65], synthesizing images by combining a filtered subset of prompts from Parti [64] with 10 representative styles from the evaluation set (see Appendix for details). The Parti subset consists of 190 prompts, each describing a subject along with its superclass to reduce semantic ambiguity (e.g., A cat, animals, in watercolor painting style).
Evaluation Metrics. Following FineStyle, we evaluate generated images using two CLIP-based metrics: S txt (CLIP Text score) and S img (CLIP Image score). S txt measures the similarity between each generated image and its corresponding text prompt to assess prompt fidelity, while S img measures the similarity between the generated image and a reference style image to assess style fidelity. However, a higher S img does not always imply better stylization quality, as excessive similarity may result from content leakage or mode collapse, as mentioned in FineStyle. To provide a more balanced evaluation, we additionally report the harmonic mean of the two scores, denoted as S harmonic (Harmonic score), which jointly reflects both prompt and style
6
Table 1. Quantitative comparison with state-of-the-art style-personalized image generation models. The symbols ↑ and ↓ indicate that higher and lower values are better, respectively. The inference time is measured as the time required to generate a single image. For tuningbased methods ( StyleDrop , DreamStyler , DB-LoRA , and B-LoRA ), we present the combined inference time, which accounts for both the tuning phase (given the reference style image) and the inference time required to produce a single output.
|Metric<br>|||Ours<br>|||IP-Adapter<br>StyleAligned<br>StyleDrop<br>DreamStyler<br>DB-LoRA<br>B-LoRA<br>CSGO<br>StyleAR<br>|||
|---|---|---|
|Training-Free<br>|||<br>Vv<br>|||<br>|<br>x<br>Vv<br>x<br>x<br>x<br>x<br>x<br>x|
|Harmonic score (_S_harmonic)↑<br>Prompt fidelity (_S_txt)↑<br>Style fidelity (_S_img)↑<br>ee|0.437<br>0.334<br>0.630<br>ee|0.433<br>0.438<br>0.386<br>0.403<br>0.420<br>0.410<br>0.421<br>0.434<br>0.302<br>0.315<br>0.273<br>0.304<br>0.323<br>0.324<br>0.318<br>0.314<br>0.763<br>0.716<br>0.657<br>0.599<br>0.602<br>0.559<br>0.623<br>0.701|
|Inference time (seconds)↓<br>ee|3.58<br>ee|10.13<br>64.58<br>520.07<br>698.98<br>342.01<br>630.42<br>15.87<br>346.68|

Style Reference	Text Prompt	Ours	IP-Adapter	StyleAligned	StyleAligned	StyleDrop	DreamStyler	DB-LoRA		B-LoRA	CSGO		StyleAR
	“A cat<br>reading a<br>book”		Ur	>		y	¥	Sie			ALUN		
5	“A horned		Cea		at		2				=		
	owl with a												
	graduation												
	cap”												
ls	“A pick-up<br>truck”			Fi		°..	ne,				eur		‘eee
pameSe	“A bench”			.		—eG,	raha	A			r=	—<br>>	.

Figure 5. Qualitative comparison with state-of-the-art style-personalized image generation models.
fidelity. Formally, it is computed as:
==> picture [166 x 25] intentionally omitted <==
6.3. Comparison with state-of-the-art stylepersonalized image generation models
To demonstrate the performance and efficiency of our model, we compare our method against eight state-ofthe-art style-personalized image generation models: StyleDrop [49], StyleAligned [18], IP-Adapter [63], DreamStyler [1], DreamBooth-LoRA (DB-LoRA) [44], B-LoRA [13], CSGO [60], StyleAR [59].
In Tab. 1, we quantitatively compare our method with state-of-the-art baselines. While StyleAligned and IPAdapter show relatively high style fidelity ( S img), they exhibit noticeably lower prompt fidelity ( S txt), indicating limited semantic alignment with the input text. As highlighted in [65], high S img scores can be misleading due to issues like content leakage or mode collapse, where the model mimics the reference style image instead of transferring style. This effect is evident in the qualitative results shown in Fig. 5, where models with high S img scores, such as StyleAligned
and IP-Adapter, frequently exhibit content leakage (first and second row). In these cases, structural details from the style reference are unintentionally transferred into the output image, leading to degraded content fidelity. These findings emphasize that high style similarity alone is insufficient to ensure faithful and semantically aligned image generation.
In contrast, DB-LoRA and B-LoRA achieve relatively high scores in prompt fidelity S txt. However, they require additional fine-tuning for each new style reference, which limits scalability in practical applications. Moreover, all training-based methods suffer from long inference times, ranging from tens to hundreds of seconds per image, due to the overhead of iterative denoising or fine-tuning. Our method, by contrast, is fully training-free, up to 195 × faster, and achieves competitive results, making it well-suited for real-time and interactive use cases. As illustrated in Fig. 5, DB-LoRA and B-LoRA tend to better preserve the semantics of the input prompt, reinforcing that S txt is a reliable indicator of semantic alignment, even when S img alone may be misleading. However, despite their strong prompt adherence, both methods tend to show relatively weaker style fidelity compared to ours, suggesting that the reference style may be only partially reflected in some cases. In contrast,
7
our method reliably preserves both the intended content and the reference style. Despite being the fastest among all methods, it still achieves a strong balance between prompt fidelity and style fidelity, underscoring its practical advantage for high-quality, real-time style-personalized generation.
6.4. Ablation study
The quantitative results in Tab. 2 highlight how each proposed component contributes to achieving a balance between style fidelity and prompt fidelity. To clearly demonstrate the effect of our PFB module, we compare two variants: a direct feature replacement strategy (REP) and our Principal Feature Blending (PFB). As shown in Tab. 2-(a), the baseline configuration attains the highest prompt fidelity ( S txt) but exhibits limited style fidelity due to the absence of explicit style modulation. In contrast, directly replacing the style feature in (b) yields the highest style fidelity ( S img), but at the expense of severe prompt degradation, indicating significant content leakage from the style reference. The SVDbased blending in (c) provides a more favorable trade-off: it mitigates the prompt-fidelity drop observed in (b) while still offering a substantial improvement in style fidelity, consistent with our observation that the dominant singular component primarily captures stylistic information. Finally, the full model in (d), which integrates both PFB and SAC, achieves the most balanced performance across all metrics, yielding the highest harmonic score ( S harmonic). This demonstrates that the proposed modules effectively complement one another, enhancing style fidelity with minimal sacrifice of prompt fidelity.
The qualitative comparison in Fig. 6 further supports these trends. In Fig. 6-(a), while the baseline produces clean and coherent images, it fails to reproduce the stylistic characteristics of the reference. Direct replacement in (b) enforces strong style transfer but also introduces unintended content elements from the reference, resulting in clear prompt mismatch. The SVD-guided variant in (c) successfully captures the intended style while retaining the target content, though its prompt adherence is still weaker than the baseline. In contrast, the full model in (d) preserves the style of the reference and simultaneously generates images that align closely with the prompt, achieving the most balanced and desirable output—consistent with the quantitative trends observed above.
6.5. User study
We conduct a user study with 30 participants (ages 20s– 50s) to further support our evaluation. Participants evaluate two key aspects: prompt and style fidelity. We selected comparison models based on their quantitative performance: StyleAligned [18] and IP-Adapter [63], which achieved the highest S img (style fidelity) scores, and DB-LoRA [44] and
Table 2. Ablation study on Principal Feature Blending (PFB) and Structural Attention Correction (SAC). REP denotes replacement using the style feature F 3[sty][. The symbol] [ ↑][indicates that higher is] better. The best and second-best results are highlighted in bold and underline, respectively.
#		Method			_S_txt ↑	_S_img ↑	_S_harmonic	↑
(a)		Infinity			0.348	0.559	0.429	
(b)		Infinity + REP	Infinity + REP		0.279	0.696	0.398	
(c)		Infinity + PFB	Infinity + PFB		0.321	0.631	0.426	
(d)		Infinity + PFB + SAC	Infinity + PFB + SAC	Infinity + PFB + SAC	0.334	0.630	0.437	
			(a)	(b)		(c)	(d)	
‘oon<br>£-|a	H<br>\v<br>ve<br>-<br>H WSS<br>i<br>NSS<br>:<br>y<br>:		lm<br>Sle<br>»<br>Peete		afin<br>®<br>2<br>fm<br>‘a<br>—	OY<br>8 a,<br>c-<br>a<br>C—		
Style Reference					Text Prompt:“A flower”			
		H					pan	
Style Reference				Text Prompt:“A dragon perched on a cliff”		“A dragon perched on a cliff”		

Figure 6. Qualitative ablation study on proposed method. (a)-(d) correspond to the component in Tab. 2.
B-LoRA [13], which achieved the highest S txt (prompt fidelity) scores. Our method achieves a clearly superior preference in prompt fidelity (35.3%) while maintaining competitive style fidelity (32.0%), compared to the other models’ scores of 4.3%, 5.0%, 26.7%, 28.7% (prompt) and 30.7%, 23.3%, 8.3%, 5.7% (style). An example of the interface is in the Supplementary material.
7. Conclusion
In this work, we introduced a training-free framework for style-personalized image generation that operates on a single reference image and leverages the efficiency of a scale-wise autoregressive model. Through a detailed stepwise analysis of the model’s generation process, we identified a pivotal feature that jointly governs content and style, and further demonstrated—via an SVD-based spectral study—that its dominant singular component captures style-specific variation. Building on these insights, we proposed two lightweight yet effective modules: Principal Feature Blending , which provides precise and interpretable style control, and Structural Attention Correction , which stabilizes structural consistency during generation. Our method achieves high performance while preserving prompt fidelity, offering a favorable balance. Quantitative and qualitative evaluations confirm that the proposed components operate as intended, enabling faithful style personalization without additional training and with significantly faster inference than existing models.
8
A Training-Free Style-Personalization via SVD-Based Feature Decomposition
Supplementary Material
A. Comprehensive analysis of our method
A.1. Additional Results for Key Step Feature Analysis
In Sec. 4-(2), we showed that replacing only the largest singular component of F 3 primarily alters stylistic attributes while largely preserving content. To further validate this observation, we extend the SVD-guided manipulation experiment by varying the number of preserved singular values.
We use the same prompt setup and intervention protocol as in the main paper: we construct 100 mixed prompt pairs ( T, T[ˆ] ), each differing in both object category and color (e.g., “A photo of a red truck” vs. “A photo of a purple cat”). For each prompt, we perform singular value decomposition F 3 = U Σ V[⊤] , and reconstruct truncated variants that retain only the top- k components:
==> picture [157 x 14] intentionally omitted <==
where Σ[(] [k][)] is constructed by preserving only the largest k singular values while zeroing out the remaining entries. We evaluate k ∈{ 1 , 2 , 4 , 8 , 16 , 32 } , and for each k , we generate SVD-guided outputs by replacing the corresponding portion of F[ˆ] 3:
==> picture [163 x 15] intentionally omitted <==
where F[ˆ] 3[res][(] [k][)] = F[ˆ] 3 − F[ˆ] 3[(] [k][)] preserves the remaining feature components.
We measure object-related and color-related CLIP similarity following the same evaluation protocol used for the k = 1 experiment in the main paper. As shown in Fig. 7, color-related similarity sharply increases at k = 1 and saturates thereafter, demonstrating that the dominant singular direction primarily captures style. In contrast, object-related similarity increases gradually as k grows, indicating that higher-rank components encode structural information.
Qualitative examples in Fig. 8 show a similar trend: the k = 1 output transfers texture and color while preserving object shape, whereas larger k values begin to alter geometry and object identity. These results further support our main finding that the first principal component of F 3 predominantly encodes style, also justifying our exponential reweighting design in the main method.
A.2. Analysis of the exponential decay rate
We further conduct an additional ablation study on the exponential decay rate α in Principal Feature Blending. As shown in Tab. 3, our method remains robust across different values of α , exhibiting only a minor trade-off between style fidelity and prompt fidelity. Fig. 9 provides a visualization
==> picture [225 x 124] intentionally omitted <==
----- Start of picture text -----<br>
Style Similarity<br>Content Similarity<br>----- End of picture text -----<br>

Figure 7. Qualitative results of SVD-guided feature replacement with varying top- k singular values. From left to right: the baseline output generated from T[ˆ] , SVD-guided outputs with k ∈ { 1 , 2 , 4 , 8 , 16 , 32 } , and the baseline output generated from T .
of how varying α controls the exponential decay of weights across singular values.
Decreasing α , which increases the influence of the higher-rank singular components, naturally elevates the risk of content leakage during style injection, resulting in a decrease in prompt fidelity. This behavior is consistent with our hypothesis that the dominant singular value predominantly encodes style-related information over the remaining components. We set α = 1 . 0 as it provides the most balanced performance.
Table 3. Additional ablation study on exponential decay rate ( α ) in Principal Feature Blending (PFB). The symbol ↑ indicates that higher is better. The best and second-best results are highlighted in bold and underline, respectively.
alpha (α)	_S_txt ↑<br>_S_img ↑<br>_S_harmonic
0.2<br>0.323<br>0.640<br>0.429<br>0.6<br>0.331<br>0.631<br>0.434<br>1.0 (ours)<br>0.334<br>0.630<br>0.437<br>2.0<br>0.334<br>0.624<br>0.435<br>5.0<br>0.335<br>0.621<br>0.435	

B. Details of the dual-stream generation mechanism
We provide a detailed description of our dual-stream generation process in Algorithm 1. Both the content path and generation path are conditioned on the same text prompt T (“<content> in <style>”) and are executed jointly within a single inference batch. Using identical conditioning prevents semantic mismatch between the two streams
9
==> picture [437 x 549] intentionally omitted <==
----- Start of picture text -----<br>
𝑇 ["] 1 2 4 8 16 32 (all) 𝑇<br>: y<br>Kia 2 oo &<br>“L wtte| oo<br><A Sonon ayJ AEGeens” Mgpsae 2apk~<a Brisom me Pugwr “<SS<br>brown spaceship orange snowflake<br>black palace blue volcano<br>amare -<br>white van purple skier<br>* PS \J - ¥ ft “<br>A S 4 “s<br>red guitar orange moose<br>»e—~4 ><br>=<br>SZ<br>yellow tiger black keyboard<br>ee +4444AS MS MS MO :*«&<br>blue river white mountain<br>SZ \ Hs ‘<br>“ YN. Po oss.”.| oof ee<br>=)<br>_ Se a mn ‘ A ; a<br>purple owl green peacock<br>1 }<br>ey bla’ alala4 ae<br>red wizard blue bunny<br>----- End of picture text -----<br>

Figure 8. Qualitative results of SVD-guided feature replacement with varying k . From left to right: the baseline output generated from T[ˆ] , SVD-guided outputs with k ∈{ 1 , 2 , 4 , 8 , 16 , 32 } , and the baseline output generated from T .
and ensures that both evolve under the same textual supervision.
The content path follows the original inference process of the pre-trained model without modification, producing a
10
==> picture [213 x 132] intentionally omitted <==
----- Start of picture text -----<br>
1.0 =0.2<br>=0.6<br>0.8 =1.0<br>=2.0<br>0.6 =5.0<br>0.4<br>0.2<br>0.0<br>0 5 10 15 20 25 30<br>Singular value index i<br>)<br>(i<br>exp<br>----- End of picture text -----<br>

Figure 9. Visualization of exponential decay rates α with respect to the singular value index i ∈{ 0 , 1 , . . . , r − 1 } .
Algorithm 1 Dual-path style-personalized image generation
Input : Style reference image I[sty] , text prompt T Output : Stylized image I[gen]
1: {Fs[sty] }[S] s =1 [←E][I][(] [I][sty][)][ # Multi-scale style features] 2: Initialize F 0[con] , F 0[gen] # Same initial condition, same prompt T
3: for s = 1 to S do
4: # (1) Dual-stream iterative update (Eq. (1), (2)) 5: Fs[con] ←M ( Fs[con] − 1 [,][ E][T][(] [T][))] 6: Fs[gen] ←M ( Fs[gen] − 1 [,][ E][T][ (] [T][))] 7: if s = 3 then 8: # (2) Principal Feature Blending (PFB) 9: F 3[gen] ← Φ( F 3[sty] ) + ( F 3[gen] − Φ( F 3[gen] )) 10: end if 11: if s ∈ S fine then 12: # (3) Structural Attention Correction (SAC) 13: Q[gen] s ← Q[con] s = WQFs[con] 14: Ks[gen] ← Ks[con] = WKFs[con] 15: end if 16: end for 17: I[gen] ← Decoder( FS[gen] ) 18: return I[gen]
sequence of features {Fs[con] }[S] s =1[, which serve as a structural] reference. Meanwhile, the generation path produces its own feature sequence {Fs[gen] }[S] s =1[, which is selectively modulated] by our proposed mechanisms (PFB, SAC). Throughout inference, the content path provides structural guidance to the generation path, enabling it to preserve spatial consistency while integrating style information from the reference style image.
C. Implementation details
C.1. Implementation setup of comparison models
We conduct extensive comparisons against existing stylepersonalized image generation methods. To ensure fair and
reproducible evaluation, all baseline models are run using publicly released implementations and their default hyperparameters, without additional tuning or prompt engineering unless explicitly required. We categorize baselines into two groups: (1) tuning-based approaches, which require style-specific fine-tuning before inference, and (2) trainingfree or pre-trained approaches, which operate directly without per-style optimization. For each method, we follow the official configurations provided in their respective repositories unless otherwise stated.
Tuning-based approaches These methods require finetuning a model for each reference style image. For each style reference, we performed style-specific fine-tuning following the official instructions of each repository, and report the total runtime consisting of both (1) training time per style and (2) inference time per image in the main paper.
B-LoRA [13]: Official implementation: https:// github.com/yardenfren1996/B-LoRA
DB-LoRA [44]: Official implementation: https:// github.com/huggingface/diffusers/tree/ main/examples/dreambooth
DreamStyler [1]: Official implementation: https:// github.com/webtoon/dreamstyler
StyleDrop [49]: Unofficial PyTorch reproduction: https://github.com/zideliu/StyleDropPyTorch
Training-free or pre-trained approaches These methods do not require additional fine-tuning per style. Instead, they operate either using a pre-trained style adapter or through direct inference-time conditioning. We evaluate all methods using their official inference settings and do not perform retraining or additional dataset-specific tuning.
IP-Adapter [63]: Official implementation: https:// github.com/tencent-ailab/IP-Adapter
StyleAligned [18]: Official implementation: https:// github.com/google/style-aligned
CSGO [60]: Official implementation: https : / / github.com/instantX-research/CSGO
StyleAR [59]: Official implementation: https:// github.com/wuyi2020/StyleAR
All models are evaluated under a unified hardware environment using a single NVIDIA A6000 GPU with PyTorch.
C.2. Styles and prompts for generation
Fig. 10 presents the style prompts for the reference images used in the paper. Images marked with * indicate those used for the quantitative evaluation, for which we use the same prompts as Finestyle [65]. The style prompts serve as a high-level guide during the generation process, allowing the model to better align visual features with the target
11
style. This provides a lightweight, training-free alternative to methods that require additional training. Note that our method does not rely on detailed prompts; simple, highlevel style categories (e.g., “oil painting,” “3d rendering,” etc.) are sufficient.
C.3. User Study Details
To complement our quantitative evaluation, we conduct a user study involving 30 participants (ages 20s–50s). Participants compare results across two criteria: prompt fidelity (semantic alignment with text) and style fidelity (visual similarity to the reference style). Each comparison presents participants with a reference style image, a target text prompt, and outputs from multiple models.
We select comparison models based on their quantitative performance: StyleAligned [18] and IP-Adapter [63], which achieved the highest S img (style fidelity), and DBLoRA [44] and B-LoRA [13], which achieved the highest S txt (prompt fidelity). This selection ensures that the user study compares the strongest-performing baselines under each metric.
As shown in Tab. 4, our method achieves the highest preference in prompt fidelity (35.3%) while maintaining competitive style fidelity (32.0%). Notably, prompt-tuned baselines (DB-LoRA, B-LoRA) exhibit strong semantic alignment but fail to preserve style, while style-focused baselines (StyleAligned, IP-Adapter) preserve style but lack semantic consistency. An example of the interface used in the study is shown in Fig. 11.
Table 4. User study preference results (percentage).
Model	Prompt Fidelity↑	Style Fidelity↑
StyleAligned [18]	4.3%	30.7%
IP-Adapter [63]	5.0%	23.3%
DB-LoRA [44]	26.7%	8.3%
B-LoRA [13]	28.7%	5.7%
Ours	35.3%	32.0%

D. Incorporating ours into other scale-wise autoregressive models
Our method is designed to be model-agnostic within the family of scale-wise autoregressive generative models, as it operates directly without modifying model weights or requiring retraining. To validate its generalization ability, we apply our method to two additional models beyond our primary backbone, Infinity-2B [16].
We first implement our method on Infinity-8B, a larger variant of our baseline model with increased capacity and parameter count. As shown in Fig. 12-(Top), our method produces consistent and stable stylization effects across this
stronger model configuration, demonstrating robustness to architectural scaling without additional tuning or adaptation. We further apply our method to Switti [56], a distinct scale-wise autoregressive text-to-image model that differs structurally from Infinity. Despite architectural differences, our plug-and-play modules function reliably without modification, producing coherent, style-personalized generations, as shown in Fig. 12-(Bottom). This result supports our approach to generalizing across models that share the scale-wise autoregressive generation paradigm.
E. Future work and limitations
Our work presents a training-free style-personalized image generation framework grounded in a comprehensive analysis of a scale-wise autoregressive model. By identifying a key step that significantly influences the output image and demonstrating that dominant singular components of its feature space effectively capture style information, we establish a principled mechanism for style extraction and injection. We believe that this analysis opens up several promising future directions, enabling more precise and flexible control over style, content, and other visual attributes in personalized image generation systems.
Despite these strengths, our method faces limitations when the style reference image contains heterogeneous or conflicting stylistic attributes (e.g., mixed artistic media or multiple visual motifs), as it lacks an explicit mechanism to disentangle and selectively transfer specific sub-styles. Since our style extraction relies on dominant singular components, the injected style may reflect a blended representation of multiple styles rather than a feature representing a single, isolated style. Future research could incorporate localized style decomposition, spatially variant basis representations, or user-guided selection to enable more finegrained style control.
F. Additional qualitative results
F.1. Additional results
Fig. 13 presents additional qualitative results demonstrating that our method faithfully transfers style-specific information from the reference image while suppressing irrelevant details, effectively avoiding content leakage or mode collapse. This enables expressive and robust style personalization that generalizes well across diverse scenes and artistic styles.
F.2. Style-aligned image generation
Furthermore, we demonstrate that our model can perform style-aligned image generation using only a style prompt, without requiring a reference style image, by including a dedicated style pathway in the same batch derived from the style text prompt and leveraging its third feature as
12
the style representation. As shown in Fig. 14, our model shows competitive performance compared to representative style-aligned image generation models [18, 66], indicating its capability in style-aligned image generation. These results validate that our method can operate effectively in both image-guided and text-guided style-related generation scenarios in a unified and training-free manner.
13
colorful origami 3d rendering
matte textured 3d rendering* watercolor painting* intricate line-art illustration geometric flat illustration
northern renaissance art - 2 _ SL) : So .-¢ ad. claymation
metallic 3D rendering
kid crayon drawing* oil painting style* a Vel Ge es | t} cartoon style illustration glowing neon*
neon splash comic art architectural line art
retro sci-fi graphic
flat cartoon vector art* matte and worn out textured wooden sculpture*
melting golden 3d oil painting style* rendering*
short line drawing*
Figure 10. Style images and corresponding prompts. The symbol * indicates those used for quantitative evaluation.
14
==> picture [62 x 9] intentionally omitted <==
----- Start of picture text -----<br>
[Prompt Fidelity]<br>----- End of picture text -----<br>

==> picture [55 x 9] intentionally omitted <==
----- Start of picture text -----<br>
[Style Fidelity]<br>----- End of picture text -----<br>

Figure 11. Example interface used in the user study. Participants selected the best-performing method among five candidates for each evaluation criterion.
15
==> picture [429 x 569] intentionally omitted <==
----- Start of picture text -----<br>
A beacon A mythical<br>flare tower A coral branch A gyroscope A campfire rune stone<br>Infinity-8B<br>Style Reference<br>—S Pee<br>Infinity-8B os: + | = Pe Ua=<br>+ [ele)] [2,] [oe]<br>82 & [Wi]<br>Ours<br>A basketball  A mythical<br>hoop A paper boat A mailbox rune stone A guitar<br>Infinity-8B [3| og] El | eR<br>Style Reference<br>Infinity-8B<br>+<br>Ours<br>’ | Ces aK<br>A transit A feather A sandtimer A helmet A bell<br>Switti<br>G=-<br>Style Reference<br>Z;<br>Switti + Ours<br>= ) » SS AF —<br>A rune-etched<br>A compass rose stone tablet A dreamcatcher A metronome A satellite dish<br>Switti eA QO hk &<br>Style Reference<br>Switti + Ours<br>----- End of picture text -----<br>

Figure 12. Qualitative results of applying our method to other scale-wise autoregressive models.
16
==> picture [474 x 499] intentionally omitted <==
----- Start of picture text -----<br>
Style Reference A pine tree line A bear A cluster of pebbles A row of mountains A pair of boots<br>¥ - @ { AA Dy,<br>Style Reference An oval mirror A leaf A lemon A moose A motorbike<br>ge “eX y : g, Lees BIOS N , ¢<br>Style Reference A deer A dune crab A mountain peak A robotic fish A drone<br>- @ ] FR.) — .<br>trie oro fends BP, \ig Gl > |CE<br>ie | (Jat) | AR? | | ee er<br>sm i 4 — ee e i. 2 y —~— aq = Zz ~ a 3<br>Style Reference A boom box A cube robot A spaceship A speeding skier A jeep car<br>Style Reference A chef cooking A phoenix A teddybear A turtle An arm chair<br>:<br>oi a =<br>Style Reference An observatory A lighthouse A taxi A dragon A domed city on cliffs<br>----- End of picture text -----<br>

Figure 13. Various style-personalized results of our model.
17
==> picture [252 x 522] intentionally omitted <==
----- Start of picture text -----<br>
Ours , 9 ‘ Af<br>StyleAligned A— a¥ : aesy 4 " x + GoBe<br>o ne~ 2, a ae<br>AlignedGen ys we fily<br>{ A dragon, A dwarf, A mushroom, An Elf } in glowing style.<br>Ours<br>Ne SS) oe ie = 7 PS a we<br>StyleAligned<br>AlignedGen mc) ay) —_—. kg a Sai<br>8 4. “4 v d ae. * ><br>{ Clock, Whale, Starfish, Helicopter } in retro poster style.<br>Ours<br>StyleAligned<br>. a bo ~ ™ =<br>ae aE ° Tk} aoa<br>AlignedGen ry eshi . ig £|4 AesaysLeg ee<br>a Rane” ms, ee 1,<br>{ An astronaut, A diver, A carousel, Bowl of fruits } in celestial<br>artwork style.<br>----- End of picture text -----<br>

Figure 14. Style-aligned image generation results with text-only style descriptions. Each row represents a different content prompt, and each column applies a distinct style, as described in the text.
18
References
[1] Namhyuk Ahn, Junsoo Lee, Chunggi Lee, Kunhee Kim, Daesik Kim, Seung-Hun Nam, and Kibeom Hong. Dreamstyler: Paint by style inversion with text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence , pages 674–681, 2024. 2, 3, 7, 11
[2] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In European conference on computer vision , pages 707–723. Springer, 2022. 2
[3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 18392–18402, 2023. 2
[4] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 2
[5] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 11315–11325, 2022. 3
[6] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704 , 2023. 1, 2, 3
[7] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 3558–3568, 2021. 2
[8] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. Journal of Machine Learning Research , 25(70):1–53, 2024. 3
[9] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo. Style injection in diffusion: A training-free approach for adapting largescale diffusion models for style transfer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 8795–8805, 2024. 2
[10] Yingying Deng, Fan Tang, Weiming Dong, Wen Sun, Feiyue Huang, and Changsheng Xu. Arbitrary style transfer via multi-adaptation network. In Proceedings of the 28th ACM international conference on multimedia , pages 2719–2727, 2020. 2
[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 , 2020. 2
[12] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Pro-
ceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 12873–12883, 2021. 3
[13] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora. In European Conference on Computer Vision , pages 181–198. Springer, 2024. 2, 3, 7, 8, 11, 12
[14] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2414–2423, 2016. 2
[15] Jongmin Gim, Jihun Park, Kyoungmin Lee, and Sunghoon Im. Content-adaptive style transfer: A training-free approach with vq autoencoders. In Proceedings of the Asian Conference on Computer Vision , pages 2337–2353, 2024. 2
[16] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 15733–15744, 2025. 1, 2, 3, 5, 6, 12
[17] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 2328–2337, 2023. 2
[18] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 4775–4785, 2024. 2, 3, 7, 8, 11, 12, 13
[19] Kibeom Hong, Seogkyu Jeon, Junsoo Lee, Namhyuk Ahn, Kunhee Kim, Pilhyeon Lee, Daesik Kim, Youngjung Uh, and Hyeran Byun. Aespa-net: Aesthetic pattern-aware style transfer networks. In Proceedings of the IEEE/CVF international conference on computer vision , pages 22758–22767, 2023. 2
[20] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision , pages 1501–1510, 2017. 2
[21] Jaeseok Jeong, Mingi Kwon, and Youngjung Uh. Trainingfree content injection using h-space in diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , pages 5151–5161, 2024. 2
[22] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 10124–10134, 2023. 2
[23] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125 , 2023. 3
[24] Gihyun Kwon and Jong Chul Ye. Clipstyler: Image style transfer with a single text condition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 18062–18071, 2022. 2
19
[25] Gihyun Kwon and Jong Chul Ye. Diffusion-based image translation using disentangled style and content representation. arXiv preprint arXiv:2209.15264 , 2022. 2
[26] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2
[27] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems , 36:30146–30166, 2023. 2, 3
[28] Yijun Li, Chen Fang, Jimei Yang, Zhaowen Wang, Xin Lu, and Ming-Hsuan Yang. Universal style transfer via feature transforms. Advances in neural information processing systems , 30, 2017. 2
[29] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 8640–8650, 2024. 2, 3
[30] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13 , pages 740–755. Springer, 2014. 2
[31] Chang Liu, Viraj Shah, Aiyu Cui, and Svetlana Lazebnik. Unziplora: Separating content and style from a single image. arXiv preprint arXiv:2412.04465 , 2024. 2
[32] Songhua Liu, Tianwei Lin, Dongliang He, Fu Li, Meiling Wang, Xin Li, Zhengxing Sun, Qian Li, and Errui Ding. Adaattn: Revisit attention mechanism in arbitrary neural style transfer. In Proceedings of the IEEE/CVF international conference on computer vision , pages 6649–6658, 2021. 2
[33] Ming Lu, Hao Zhao, Anbang Yao, Yurong Chen, Feng Xu, and Li Zhang. A closed-form solution to universal style transfer. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 5952–5961, 2019. 2
[34] Quang-Binh Nguyen, Minh Luu, Quang Nguyen, Anh Tran, and Khoi Nguyen. Csd-var: Content-style decomposition in visual autoregressive models. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 17013–17023, 2025. 4
[35] Dae Young Park and Kwang Hee Lee. Arbitrary style transfer with style-attentional networks. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 5880–5888, 2019. 2
[36] Jihun Park, Jongmin Gim, Kyoungmin Lee, Seunghun Lee, and Sunghoon Im. Style-editor: Text-driven object-centric style editing. arXiv preprint arXiv:2408.08461 , 2025. 2
[37] Jihun Park, Jongmin Gim, Kyoungmin Lee, Minseok Oh, Minwoo Choi, Jaeyeul Kim, Woo Chool Park, and Sunghoon Im. A training-free style-aligned image generation with scale-wise autoregressive model. arXiv preprint arXiv:2504.06144 , 2025. 3, 4
[38] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 , 2023. 1, 2
[39] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations , 2023. 2
[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pages 8748–8763. PmLR, 2021. 2, 3
[41] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning , pages 8821–8831. Pmlr, 2021. 1, 2
[42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 10684–10695, 2022. 1, 2
[43] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 22500– 22510, 2023. 2, 3
[44] Simo Ryu. Low-rank adaptation for fast text-toimage diffusion fine-tuning, 2022. URL https://github. com/cloneofsimo/lora , 10:19, 2022. 2, 3, 7, 8, 11, 12
[45] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems , 35:36479–36494, 2022. 1, 2
[46] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems , 35:25278–25294, 2022. 2
[47] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. In European Conference on Computer Vision , pages 422–438. Springer, 2024. 2, 3
[48] Joonghyuk Shin, Alchan Hwang, Yujin Kim, Daneul Kim, and Jaesik Park. Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) , 2025. 6
[49] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983 , 2023. 2, 3, 7, 11
[50] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and
20
Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer. arXiv preprint arXiv:2410.10812 , 2024. 1, 2, 3
[51] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905 , 2024. 3
[52] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 1921–1930, 2023. 2
[53] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 1921–1930, 2023. 6
image diffusion models. arXiv preprint arXiv:2308.06721 , 2023. 7, 8, 11, 12
[64] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789 , 2(3):5, 2022. 6
[65] Gong Zhang, Kihyuk Sohn, Meera Hahn, Humphrey Shi, and Irfan Essa. Finestyle: Fine-grained controllable style personalization for text-to-image models. Advances in Neural Information Processing Systems , 37:52937–52961, 2024. 2, 3, 6, 7, 11
[66] Jiexuan Zhang, Yiheng Du, Qian Wang, Weiqi Li, Yu Gu, and Jian Zhang. Alignedgen: Aligning style across generated images. arXiv preprint arXiv:2509.17088 , 2025. 13
[54] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems , 30, 2017. 3
[55] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017. 2
[56] Anton Voronov, Denis Kuznedelev, Mikhail Khoroshikh, Valentin Khrulkov, and Dmitry Baranchuk. Switti: Designing scale-wise transformers for text-to-image synthesis. arXiv preprint arXiv:2412.01819 , 2024. 12
[57] Anton Voronov, Denis Kuznedelev, Mikhail Khoroshikh, Valentin Khrulkov, and Dmitry Baranchuk. Switti: Designing scale-wise transformers for text-to-image synthesis, 2025. 3
[58] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 15943–15953, 2023. 2, 3
[59] Yi Wu, Lingting Zhu, Shengju Qian, Lei Liu, Wandi Qiao, Lequan Yu, and Bin Li. Stylear: Customizing multimodal autoregressive model for style-aligned text-to-image generation. arXiv preprint arXiv:2505.19874 , 2025. 7, 11
[60] Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, and Zechao Li. Csgo: Contentstyle composition in text-to-image generation. arXiv preprint arXiv:2408.16766 , 2024. 7, 11
[61] Serin Yang, Hyunmin Hwang, and Jong Chul Ye. Zero-shot contrastive loss for text-guided diffusion image style transfer. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 22873–22882, 2023. 2
[62] Yuan Yao, Jianqiang Ren, Xuansong Xie, Weidong Liu, Yong-Jin Liu, and Jun Wang. Attention-aware multi-stroke style transfer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 1467– 1475, 2019. 2
[63] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-to-
21