"""
Helper functions for relevance checking functionality.
This module contains CLIP classification, LLaVA analysis, and prompt building utilities.

IMPORTANT: All functions in this module work with extracted frames (numpy arrays):
- Single frame for images: shape (H, W, C)
- Multiple frames for videos: shape (N, H, W, C)
"""

import os
import time
import json
import re
import tempfile
import random
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPModel, CLIPProcessor

# LLaVA via Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama library not available. Install with: pip install ollama")


class RelevanceCheckHelpers:
    """Helper class containing utility functions for relevance checking."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.embedding_dim = config.get('embedding_dim', 512)
        self.llava_model = config.get('llava_model', 'llava-phi3:3.8b')
        
        # Initialize CLIP models if available
        self._init_clip_models()
    
    def _init_clip_models(self):
        """Initialize CLIP models from config."""
        if self.config.get('clip_model_id'):
            print("*****************************************"+self.config.get('clip_model_id'))
            try:
                # Vision-only encoder
                # self.clip_processor = CLIPImageProcessor.from_pretrained(self.config['clip_model_id'])
                # self.clip_vision = CLIPVisionModel.from_pretrained(
                #     self.config['clip_model_id'],
                #     torch_dtype=self.dtype
                # )
                # self.clip_vision = self.clip_vision.to(self.device).eval()
                
                # Full CLIP for text-image similarity
                self.clip_zero_processor = CLIPProcessor.from_pretrained(self.config['clip_model_id'])
                self.clip_model = CLIPModel.from_pretrained(
                    self.config['clip_model_id'], 
                    torch_dtype=self.dtype
                )
                self.clip_model = self.clip_model.to(self.device).eval()
                print("✅ CLIP models initialized for relevance check")
            except Exception as e:
                print(f"⚠️ CLIP models not initialized: {e}")
                self.clip_processor = None
                self.clip_vision = None
                self.clip_zero_processor = None
                self.clip_model = None
        else:
            self.clip_processor = None
            self.clip_vision = None
            self.clip_zero_processor = None
            self.clip_model = None
    
    def _ensure_embedding_dim(self, embedding_vec: np.ndarray) -> np.ndarray:
        """Ensure embedding matches configured dimension by trimming or padding"""
        if embedding_vec.ndim != 1:
            embedding_vec = embedding_vec.flatten()
        current_dim = embedding_vec.shape[0]
        if current_dim == self.embedding_dim:
            return embedding_vec.astype(np.float32, copy=False)
        if current_dim > self.embedding_dim:
            return embedding_vec[:self.embedding_dim].astype(np.float32, copy=False)
        # Pad with zeros if smaller
        padded = np.zeros(self.embedding_dim, dtype=np.float32)
        padded[:current_dim] = embedding_vec[:current_dim]
        return padded
    
    def _prepare_video_frames_for_clip(self, frames: np.ndarray, desired_frames: int = 8) -> np.ndarray:
        """Ensure we have exactly `desired_frames` frames for CLIP video analysis."""
        if frames.size == 0:
            return np.array([])
        if len(frames.shape) == 3:
            frames_arr = frames[np.newaxis, ...]
        else:
            frames_arr = frames

        if len(frames_arr.shape) != 4 or frames_arr.shape[0] == 0:
            return np.array([])

        total_frames = frames_arr.shape[0]
        if total_frames < desired_frames:
            last_frame = frames_arr[-1]
            pad_count = desired_frames - total_frames
            pad_frames = np.repeat(last_frame[np.newaxis, ...], pad_count, axis=0)
            frames_arr = np.concatenate([frames_arr, pad_frames], axis=0)
        elif total_frames > desired_frames:
            idxs = np.linspace(0, total_frames - 1, desired_frames).astype(int)
            frames_arr = frames_arr[idxs]

        return frames_arr
    
    def _clip_classify_frame(self, frame: np.ndarray, positive_text: str, negative_text: str) -> Tuple[float, np.ndarray]:
        """
        Run CLIP positive/negative classification on a single frame.
        Returns (positive_probability, normalized_embedding)
        """
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        pil_image = Image.fromarray(frame.astype(np.uint8))

        if hasattr(self, 'clip_zero_processor') and hasattr(self, 'clip_model') and self.clip_zero_processor and self.clip_model:
            try:
                inputs = self.clip_zero_processor(
                    text=[positive_text, negative_text],
                    images=pil_image,
                    return_tensors="pt",
                    padding=True
                )
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.to(self.device)

                with torch.inference_mode():
                    outputs = self.clip_model(**inputs)
                    logits = getattr(outputs, "logits_per_image", None)
                    if logits is not None:
                        probs = torch.softmax(logits, dim=1)
                        positive_prob = float(probs[0, 0].detach().cpu().item())
                    else:
                        positive_prob = 0.5

                    image_embeds = getattr(outputs, "image_embeds", None)
                    if image_embeds is not None:
                        embedding = image_embeds.squeeze(0).detach().cpu().numpy()
                    elif hasattr(self.clip_model, "get_image_features"):
                        image_inputs = self.clip_zero_processor(images=pil_image, return_tensors="pt")
                        for key, value in image_inputs.items():
                            if isinstance(value, torch.Tensor):
                                image_inputs[key] = value.to(self.device)
                        image_features = self.clip_model.get_image_features(**image_inputs)
                        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                        embedding = image_features.squeeze(0).detach().cpu().numpy()
            except Exception as e:
                print(f"⚠️ CLIP classification error: {e}")
                positive_prob = 0.5
            else:
                embedding = self._ensure_embedding_dim(embedding)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return positive_prob, embedding.astype(np.float32, copy=False)

        # Fallback: only embedding available
        embedding = self._generate_clip_embedding(frame, file_type='image')
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return 0.5, embedding.astype(np.float32, copy=False)
    
    def _clip_classify_frames_batch(self, frames: np.ndarray, positive_text: str, negative_text: str) -> List[Tuple[float, np.ndarray]]:
        """
        Run CLIP positive/negative classification on multiple frames in a single batch.
        This is much faster than processing frames sequentially.
        Returns list of (positive_probability, normalized_embedding) tuples.
        """
        if not hasattr(self, 'clip_zero_processor') or not hasattr(self, 'clip_model') or not self.clip_zero_processor or not self.clip_model:
            # Fallback to single frame processing
            results = []
            for frame in frames:
                prob, emb = self._clip_classify_frame(frame, positive_text, negative_text)
                results.append((prob, emb))
            return results

        try:
            # Convert all frames to PIL Images
            pil_images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
            
            # Process all frames in a single batch
            inputs = self.clip_zero_processor(
                text=[positive_text, negative_text],
                images=pil_images,
                return_tensors="pt",
                padding=True
            )
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.device)

            with torch.inference_mode():
                outputs = self.clip_model(**inputs)
                logits = getattr(outputs, "logits_per_image", None)
                
                if logits is not None:
                    # logits shape: [num_images, 2] (2 = positive, negative)
                    probs = torch.softmax(logits, dim=1)
                    positive_probs = probs[:, 0].detach().cpu().numpy()  # Get positive probabilities
                else:
                    positive_probs = np.full(len(frames), 0.5)

                # Get embeddings for all frames
                image_embeds = getattr(outputs, "image_embeds", None)
                if image_embeds is not None:
                    embeddings = image_embeds.detach().cpu().numpy()
                elif hasattr(self.clip_model, "get_image_features"):
                    # Fallback: get image features separately
                    image_inputs = self.clip_zero_processor(images=pil_images, return_tensors="pt")
                    for key, value in image_inputs.items():
                        if isinstance(value, torch.Tensor):
                            image_inputs[key] = value.to(self.device)
                    image_features = self.clip_model.get_image_features(**image_inputs)
                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                    embeddings = image_features.detach().cpu().numpy()
                else:
                    embeddings = np.zeros((len(frames), self.embedding_dim), dtype=np.float32)

            # Normalize and format results
            results = []
            for i in range(len(frames)):
                embedding = self._ensure_embedding_dim(embeddings[i])
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                results.append((float(positive_probs[i]), embedding.astype(np.float32, copy=False)))
            
            return results

        except Exception as e:
            print(f"⚠️ CLIP batch classification error: {e}, falling back to sequential processing")
            # Fallback to single frame processing
            results = []
            for frame in frames:
                prob, emb = self._clip_classify_frame(frame, positive_text, negative_text)
                results.append((prob, emb))
            return results
    
    def _generate_clip_embedding(self, frames: np.ndarray, file_type: str) -> np.ndarray:
        """Generate CLIP embedding (512-dim) for redundancy checking"""
        try:
            if file_type == 'video':
                frame = frames[0] if len(frames.shape) == 4 else frames
            else:
                frame = frames if len(frames.shape) == 3 else frames[0]
            
            pil_image = Image.fromarray(frame.astype(np.uint8))
            
            # Prefer CLIPModel (includes projection to 512 dims)
            if hasattr(self, 'clip_zero_processor') and hasattr(self, 'clip_model') and self.clip_zero_processor and self.clip_model:
                inputs = self.clip_zero_processor(images=pil_image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
                with torch.inference_mode():
                    image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                    embedding_np = image_features.squeeze(0).to(torch.float32).cpu().numpy()
                return self._ensure_embedding_dim(embedding_np)
            
            # Fallback to vision encoder if full CLIP model unavailable
            if hasattr(self, 'clip_processor') and hasattr(self, 'clip_vision') and self.clip_processor and self.clip_vision:
                inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(self.device, dtype=self.dtype)
                with torch.inference_mode():
                    outputs = self.clip_vision(**inputs)
                    embedding = torch.nn.functional.normalize(outputs.pooler_output, p=2, dim=-1)
                    embedding_np = embedding.squeeze(0).to(torch.float32).cpu().numpy()
                return self._ensure_embedding_dim(embedding_np)
            
            # Fallback: zero embedding
            return np.zeros(self.embedding_dim, dtype=np.float32)
        except Exception as e:
            print(f"  ⚠️ CLIP embedding generation error: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _build_vision_prompt_image(self) -> str:
        """Stage 1 for IMAGE: Comprehensive object detection - NO HALLUCINATION"""
        return """Describe ONLY what you clearly see in this image. Be factual and comprehensive.

List ALL objects visible in the image:
- Physical objects (pencil, paper, basket, hat, etc.)
- Furniture (desk, chair, etc.)
- People or animals
- Text or labels you can read

Be specific: say "pencil" not "object", say "supplies basket" not "basket".

Do not guess or imagine things that are not there.

If this is a map, chart, screenshot, or graphic - say so clearly."""
    
    def _build_logic_prompt_image(self, task_description: str, image_description: str, restrictions: Optional[str] = None) -> str:
        """Stage 2 for IMAGE: Crowdsourcing Task-Object Matcher"""
        restrictions_section = ""
        if restrictions and restrictions.lower() not in ["none", "not specified", ""]:
            restrictions_section = f"""

- RESTRICTIONS (CRITICAL - MUST ENFORCE):
{restrictions}

IMPORTANT RESTRICTION RULES:
- If EXCLUDE items are present in the image → REJECT immediately (match=false)
- If REQUIRE_ONLY is specified and image contains other types → REJECT (match=false)
- If ACCURACY requirements are not met → REJECT (match=false)
"""
        
        return f"""You are a Crowdsourcing Task-Object Matcher. Your role is to validate if the visual content contains the required objects from the task description.

INPUT DATA:

- Task Request: "{task_description}"
{restrictions_section}
- Visual Evidence (Image Report): "{image_description}"

YOUR MISSION:

Determine if the Visual Evidence contains the objects/subjects required by the Task Request.

LOGIC PROTOCOL:

1. DECONSTRUCT THE TASK: 

   - Identify the Core Object/Subject required (e.g., "office supplies", "a cat", "a receipt").

   - Identify Negative Constraints (e.g., "no screens", "no drawings", "reject if X").

2. EVALUATE EVIDENCE:

   - Does the evidence contain the Core Object/Subject from the task?

   - Is the semantic category correct? (e.g., Task: "office supplies" -> Evidence: "pencil, paper" = MATCH).

   - Are negative constraints violated? (e.g., Task says "reject if screen" and evidence shows "monitor" -> REJECT).

3. CATEGORY RELATIONSHIPS:

   - "office supplies" includes: pencil, pen, paper, notebook, stapler, supplies basket, etc.
   - "food" includes: sandwich, burger, pizza, fruit, vegetables, etc.
   - "vehicle" includes: car, bus, truck, motorcycle, bicycle, etc.
   - "animal" includes: dog, cat, bird, fish, any living creature
   - "screen/display" includes: TV, monitor, laptop screen, phone screen, tablet, etc.

4. EDGE CASE HANDLING:

   - Screen/Photo-of-Screen: If the task asks for physical objects but the evidence is a screenshot/map/chart -> REJECT (unless task specifies digital content).

   - Text Match: If the task requires specific text (e.g., "Store Name"), check if that text is visible in the evidence.

DECISION MATRIX:

- MATCH = TRUE if the required objects/subjects are present and no negative constraints are violated.

- MATCH = FALSE if the required objects are missing, wrong category, or negative constraints are violated.

OUTPUT FORMAT (JSON):

{{
  "match": boolean, 
  "confidence_score": float (0.0 to 1.0),
  "primary_observation": "One sentence summary of what was found.",
  "reasoning": "Step-by-step logic. Why does X match/fail Y?",
  "rejection_reason": "null" or "Category Mismatch" / "Constraint Violation" / "Object Not Found"
}}"""
    
    def _build_vision_prompt_video(self) -> str:
        """Stage 1 for VIDEO: Precision object detection"""
        return """
    ROLE: You are a Precision Object Detector.
    TASK: List every physical object visible in the image.
    
    OUTPUT REQUIREMENTS:
    1. Group objects by category (Electronics, Furniture, People, Vehicles, Animals).
    2. Be specific (e.g., use "Bus" instead of "Vehicle").
    3. Output a simple text list.
    """
    
    def _build_logic_prompt_video(self, task_description: str, image_description: str, restrictions: Optional[str] = None) -> str:
        """Stage 2 for VIDEO: Few-shot logic check"""
        restrictions_section = ""
        if restrictions and restrictions.lower() not in ["none", "not specified", ""]:
            restrictions_section = f"""
    
    ### RESTRICTIONS (CRITICAL - MUST ENFORCE):
    {restrictions}
    
    RESTRICTION RULES:
    - If EXCLUDE items are present → REJECT immediately (match=false)
    - If REQUIRE_ONLY is specified and evidence contains other types → REJECT (match=false)
    - If ACCURACY requirements are not met → REJECT (match=false)
"""
        
        return f"""
    ### SYSTEM ROLE
    You are a Semantic Data Auditor. Your job is to validate if visual evidence matches a user task.
{restrictions_section}
    ### FEW-SHOT EXAMPLES (LEARN FROM THESE)
    
    EXAMPLE 1 (Direct Synonym Match):
    User Task: "Show me a screen or TV"
    Evidence: "I see a Dell Computer Monitor on a desk."
    Output: {{ "match": true, "element_analysis": {{"Monitor": "RELATED - Monitor is a type of screen"}}, "reason": "Monitor is a type of screen." }}

    EXAMPLE 2 (Negative Constraint - Item Present):
    User Task: "Show a person. REJECT if wearing a hat."
    Evidence: "A man is standing wearing a baseball cap."
    Output: {{ "match": false, "element_analysis": {{"man": "RELATED - person detected", "baseball cap": "VIOLATION - hat detected, must reject"}}, "reason": "Constraint violation: Hat detected." }}

    EXAMPLE 3 (Negative Constraint - Item NOT Present = GOOD):
    User Task: "provide video of wildlife animals, reject any content that includes a giraffe"
    Evidence: "I see 5 elephants walking near a water hole."
    Output: {{ "match": true, "element_analysis": {{"elephants": "RELATED - elephants are wildlife animals", "water hole": "NEUTRAL - background element"}}, "reason": "Elephants are wildlife animals and no giraffe is present." }}

    EXAMPLE 4 (Public Transport):
    User Task: "provide video of public transport, reject any content that includes a private car"
    Evidence: "I see a bus and several people waiting at a bus stop."
    Output: {{ "match": true, "element_analysis": {{"bus": "RELATED - bus is public transport", "people": "NEUTRAL - passengers", "bus stop": "RELATED - public transport infrastructure"}}, "reason": "Bus is public transport and no private car detected." }}

    ### YOUR CURRENT TASK
    User Task: "{task_description}"
    Visual Evidence: "{image_description}"

    ### INSTRUCTIONS (FOLLOW STEP BY STEP)
    1. ELEMENT ANALYSIS: For EACH element in the Visual Evidence, ask yourself:
       - "Is this element related to the task?" (RELATED/NEUTRAL/UNRELATED)
       - "Does this element violate any constraint?" (VIOLATION if yes)
    2. EXPAND SYNONYMS: If task mentions "public transport", look for Bus, Train, Tram, Metro, Subway.
    3. CHECK CONSTRAINTS: Look for "Reject" words in the task. 
       - If a rejected item is NOT present in the evidence, this is GOOD (no violation).
       - Only mark VIOLATION if the rejected item IS actually present.
    4. FINAL DECISION: 
       - match=true if at least one RELATED element exists AND no VIOLATION found
       - match=false if no RELATED element OR any VIOLATION found
    
    Output JSON with element_analysis, match, and reason:
    """
    
    def _clean_json_string(self, json_str: str) -> str:
        """Extract valid JSON from the model's response, handling markdown and extra text."""
        # Remove markdown code blocks
        json_str = re.sub(r'```json', '', json_str)
        json_str = re.sub(r'```', '', json_str)
        
        # Find the curly braces
        start = json_str.find('{')
        end = json_str.rfind('}')
        
        if start != -1 and end != -1:
            return json_str[start:end+1]
        return json_str
    
    def _llava_analyze_frame(self, frame: np.ndarray, task_description: str, is_video: bool = False, restrictions: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a single frame with LLaVA using two-stage approach:
        Stage 1: Blind Vision Analysis - Describe what's in the image
        Stage 2: Logic & Constraint Check - Apply task to description with few-shot examples
        """
        if not OLLAMA_AVAILABLE:
            return {
                "final_decision": "REJECT",
                "match_score": 0.0,
                "requirement_met": False,
                "reasoning": "Ollama library not available"
            }
        
        try:
            # Convert numpy array to PIL Image
            if len(frame.shape) == 3:
                pil_image = Image.fromarray(frame.astype(np.uint8))
            else:
                pil_image = Image.fromarray(frame[0].astype(np.uint8))
            
            # Save to temporary file for Ollama
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                pil_image.save(tmp_file.name, 'JPEG')
                tmp_path = tmp_file.name
            
            try:
                # ==========================================
                # STAGE 1: BLIND VISION ANALYSIS
                # ==========================================
                print(f"  - Stage 1: Vision Analysis...")
                # Use different prompts for video vs image
                if is_video:
                    vision_prompt = self._build_vision_prompt_video()
                else:
                    vision_prompt = self._build_vision_prompt_image()
                
                # Call LLaVA for vision analysis (with image)
                vision_response = ollama.chat(
                    model=self.llava_model,
                    messages=[{
                        'role': 'user',
                        'content': vision_prompt,
                        'images': [tmp_path]
                    }]
                )
                
                image_description = vision_response['message']['content'].strip()
                print(f"    Model Saw: {image_description[:100]}...")
                
                # ==========================================
                # STAGE 2: LOGIC & CONSTRAINT CHECK
                # ==========================================
                print(f"  - Stage 2: Logic Check...")
                # Use different prompts for video vs image
                if is_video:
                    logic_prompt = self._build_logic_prompt_video(task_description, image_description, restrictions)
                else:
                    logic_prompt = self._build_logic_prompt_image(task_description, image_description, restrictions)
                
                # Call LLaVA for logic check (text only, no image needed)
                logic_response = ollama.chat(
                    model=self.llava_model,
                    format='json',
                    messages=[{
                        'role': 'user',
                        'content': logic_prompt,
                        'images': [tmp_path],
                    }]
                )
                
                raw_json = logic_response['message']['content']
                clean_json = self._clean_json_string(raw_json)
                
                # Parse JSON response
                try:
                    result_json = json.loads(clean_json)
                    # Ensure result_json is a dict (model might return a list)
                    if not isinstance(result_json, dict):
                        print(f"    ⚠️ JSON is not a dict, got {type(result_json).__name__}")
                        result_json = {}
                    match_bool = result_json.get("match", False)
                    
                    # For image: extract new fields from crowdsourcing QA format
                    if not is_video:
                        confidence_score = result_json.get("confidence_score", 0.0)
                        primary_observation = result_json.get("primary_observation", "")
                        reasoning_raw = result_json.get("reasoning", "No reason provided")
                        rejection_reason = result_json.get("rejection_reason", None)
                        # Use reasoning as reason_str, combine with primary_observation if available
                        reason_str = reasoning_raw
                        if primary_observation:
                            reason_str = f"{primary_observation}. {reasoning_raw}"
                        if rejection_reason:
                            reason_str = f"{reason_str} [Rejection: {rejection_reason}]"
                    else:
                        # For video: use old format
                        reason_raw = result_json.get("reason", "No reason provided")
                        reason_str = str(reason_raw) if not isinstance(reason_raw, str) else reason_raw
                        confidence_score = None
                        primary_observation = ""
                        rejection_reason = None
                    
                    # Ensure reason is always a string
                    reason_str = str(reason_str) if not isinstance(reason_str, str) else reason_str
                    
                    # For video, extract element_analysis
                    element_analysis = result_json.get("element_analysis", {})
                    if not isinstance(element_analysis, dict):
                        element_analysis = {}
                except (json.JSONDecodeError, Exception) as e:
                    print(f"    ⚠️ JSON parse error ({type(e).__name__}: {e}), falling back to REJECT")
                    match_bool = False
                    reason_str = f"JSON Parse Error: {str(e)}"
                    element_analysis = {}
                    confidence_score = None
                    primary_observation = ""
                    rejection_reason = None
                
                # Convert to standard result format
                final_decision = "ACCEPT" if match_bool else "REJECT"
                # Use confidence_score if available (for images), otherwise use standard scores
                if not is_video and confidence_score is not None:
                    match_score = confidence_score * 100.0  # Convert 0.0-1.0 to 0-100
                else:
                    match_score = 85.0 if match_bool else 15.0  # Standard scores for matched/unmatched
                
                # Extract visible elements
                visible_elements = []
                if is_video and element_analysis:
                    # For video: use element_analysis keys
                    visible_elements = [str(k) for k in element_analysis.keys()][:10]
                else:
                    # For image: extract from description
                    desc_words = image_description.split()[:10]
                    visible_elements = [w.strip('.,;:') for w in desc_words if len(w.strip('.,;:')) > 3][:5]
                
                result = {
                    "visible_elements": visible_elements,
                    "scene_description": image_description,
                    "requirement_met": match_bool,
                    "explanation": reason_str,
                    "match_score": match_score,
                    "final_decision": final_decision,
                    "classification": "video_fewshot" if is_video else "image_crowdsourcing_qa",
                    "reasoning": reason_str,
                    "element_analysis": element_analysis if is_video else {},
                    "raw_vision_response": image_description,
                    "raw_logic_response": raw_json
                }
                
                # Add crowdsourcing QA fields for images
                if not is_video:
                    result["confidence_score"] = confidence_score if confidence_score is not None else (match_score / 100.0)
                    result["primary_observation"] = primary_observation if primary_observation else ""
                    result["rejection_reason"] = rejection_reason if rejection_reason else None
                
                # Safely truncate reason for display
                display_reason = reason_str[:50] if len(reason_str) > 50 else reason_str
                print(f"    Decision: {final_decision} | Reason: {display_reason}...")
                return result
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            print(f"  ❌ LLaVA two-stage analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "visible_elements": [],
                "scene_description": "",
                "requirement_met": False,
                "explanation": f"Error: {str(e)}",
                "match_score": 0.0,
                "final_decision": "REJECT",
                "classification": "error",
                "reasoning": f"Error: {str(e)}"
            }
    
    def _llava_analyze_frames_parallel(self, frames: np.ndarray, task_description: str, is_video: bool = False, max_workers: Optional[int] = None, restrictions: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple frames in parallel using ThreadPoolExecutor.
        This significantly speeds up video processing by running LLaVA calls concurrently.
        """
        if not OLLAMA_AVAILABLE:
            return [{
                "final_decision": "REJECT",
                "match_score": 0.0,
                "requirement_met": False,
                "reasoning": "Ollama library not available"
            }] * len(frames)
        
        # Get max_workers from config or use default
        if max_workers is None:
            max_workers = self.config.get('llava_max_workers', 2)
        
        # Limit workers to avoid overwhelming the system
        num_workers = min(max_workers, len(frames), 8)  # Increased max to 8 for better parallelism
        
        # Process frames in parallel
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_frame = {
                executor.submit(self._llava_analyze_frame, frame, task_description, is_video, restrictions): idx
                for idx, frame in enumerate(frames)
            }
            
            # Collect results in order
            frame_results = [None] * len(frames)
            for future in as_completed(future_to_frame):
                frame_idx = future_to_frame[future]
                try:
                    result = future.result()
                    frame_results[frame_idx] = result
                except Exception as e:
                    print(f"  ⚠️ Error processing frame {frame_idx + 1}: {e}")
                    frame_results[frame_idx] = {
                        "final_decision": "REJECT",
                        "match_score": 0.0,
                        "requirement_met": False,
                        "reasoning": f"Error: {str(e)}"
                    }
            
            results = frame_results
        
        return results
    
    def save_llava_summary(self, summary_text: str) -> str:
        """Persist the latest LLaVA relevance summary to disk for downstream consumers"""
        logs_dir = os.path.join("output", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        summary_path = os.path.join(logs_dir, "llava_relevance_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"📝 Saved LLaVA relevance summary to {summary_path}")
        return summary_path

