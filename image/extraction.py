"""
Image Feature Extraction Pipeline

This module handles:
1. Loading images from ads/images directory
2. Standardizing dimensions and quality
3. Managing processed image database
4. Running all feature extraction methods on standardized images
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import time

import numpy as np
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for each processed image"""
    image_id: str
    original_path: str
    original_size: Tuple[int, int]  # (width, height)
    standardized_size: Tuple[int, int]
    file_size_bytes: int
    format: str
    processing_time_ms: float


@dataclass
class ExtractedFeatures:
    """Container for all extracted features from an image"""
    image_id: str
    metadata: ImageMetadata
    
    # Feature extraction results (to be filled by specialized modules)
    attention_map: Optional[Dict[str, Any]] = None
    characterization: Optional[Dict[str, Any]] = None
    emotional_tone: Optional[Dict[str, Any]] = None
    text_branding: Optional[Dict[str, Any]] = None
    visual_semantics: Optional[Dict[str, Any]] = None
    llm_description: Optional[Dict[str, Any]] = None
    
    processing_time_total_ms: float = 0.0


class ImageDatabase:
    """Manages standardized images and their metadata"""
    
    def __init__(self, cache_dir: str = "processed_images"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata: Dict[str, ImageMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing metadata if available"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                self.metadata = {
                    k: ImageMetadata(**v) for k, v in data.items()
                }
            logger.info(f"Loaded metadata for {len(self.metadata)} images")
    
    def save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(
                {k: asdict(v) for k, v in self.metadata.items()},
                f,
                indent=2
            )
    
    def get_image_path(self, image_id: str) -> Path:
        """Get path to standardized image"""
        return self.cache_dir / f"{image_id}.npy"
    
    def save_image(self, image_id: str, image_array: np.ndarray, metadata: ImageMetadata):
        """Save standardized image and metadata"""
        np.save(self.get_image_path(image_id), image_array)
        self.metadata[image_id] = metadata
    
    def load_image(self, image_id: str) -> Optional[np.ndarray]:
        """Load standardized image"""
        image_path = self.get_image_path(image_id)
        if image_path.exists():
            return np.load(image_path)
        return None
    
    def has_image(self, image_id: str) -> bool:
        """Check if image is already processed"""
        return self.get_image_path(image_id).exists()


class ImageStandardizer:
    """Standardizes image dimensions and quality"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 1024),
        maintain_aspect_ratio: bool = True,
        interpolation: int = cv2.INTER_LANCZOS4
    ):
        self.target_size = target_size
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.interpolation = interpolation
    
    def standardize(self, image_path: str) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Standardize an image to consistent dimensions and quality
        
        Returns:
            Tuple of (standardized_image_array, metadata)
        """
        start_time = time.time()
        
        # Extract image ID from filename
        image_id = Path(image_path).stem
        
        # Load image
        img = Image.open(image_path)
        original_size = img.size  # (width, height)
        original_format = img.format
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Resize image
        if self.maintain_aspect_ratio:
            img_array = self._resize_with_aspect_ratio(img_array)
        else:
            img_array = cv2.resize(
                img_array,
                self.target_size,
                interpolation=self.interpolation
            )
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        processing_time = (time.time() - start_time) * 1000  # milliseconds
        
        # Create metadata
        metadata = ImageMetadata(
            image_id=image_id,
            original_path=image_path,
            original_size=original_size,
            standardized_size=img_array.shape[:2],  # (height, width)
            file_size_bytes=os.path.getsize(image_path),
            format=original_format or 'unknown',
            processing_time_ms=processing_time
        )
        
        return img_array, metadata
    
    def _resize_with_aspect_ratio(self, img_array: np.ndarray) -> np.ndarray:
        """Resize image while maintaining aspect ratio with padding"""
        h, w = img_array.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(
            img_array,
            (new_w, new_h),
            interpolation=self.interpolation
        )
        
        # Create padded image (black padding)
        padded = np.zeros((target_h, target_w, 3), dtype=img_array.dtype)
        
        # Center the image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded


class FeatureExtractor:
    """Orchestrates all feature extraction methods"""
    
    def __init__(self, database: ImageDatabase):
        self.database = database
        
        # Import feature extraction modules (with graceful handling if not implemented)
        self.extractors = {}
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize all feature extraction modules"""
        from image import (
            attention_map,
            characterization,
            emotional_tone,
            text_branding,
            visual_semantics,
            llm_description
        )
        
        # Check which extractors have an 'extract' function
        modules = {
            'attention_map': attention_map,
            'characterization': characterization,
            'emotional_tone': emotional_tone,
            'text_branding': text_branding,
            'visual_semantics': visual_semantics,
            'llm_description': llm_description
        }
        
        for name, module in modules.items():
            if hasattr(module, 'extract'):
                self.extractors[name] = module.extract
                logger.info(f"Loaded extractor: {name}")
            else:
                logger.warning(f"Extractor not implemented: {name}")
    
    def extract_features(self, image_id: str, image_array: np.ndarray) -> ExtractedFeatures:
        """Extract all features from a standardized image"""
        start_time = time.time()
        
        # Get metadata
        metadata = self.database.metadata[image_id]
        
        # Initialize feature container
        features = ExtractedFeatures(
            image_id=image_id,
            metadata=metadata
        )
        
        # Run each extractor
        for name, extractor_func in self.extractors.items():
            try:
                logger.info(f"Running {name} on {image_id}")
                result = extractor_func(image_array, image_id)
                setattr(features, name, result)
            except Exception as e:
                logger.error(f"Error in {name} for {image_id}: {str(e)}")
                setattr(features, name, {"error": str(e)})
        
        features.processing_time_total_ms = (time.time() - start_time) * 1000
        
        return features


class ExtractionPipeline:
    """Main pipeline for image processing and feature extraction"""
    
    def __init__(
        self,
        images_dir: str = "ads/images",
        cache_dir: str = "processed_images",
        output_dir: str = "extracted_features",
        target_size: Tuple[int, int] = (1024, 1024),
        max_workers: int = 4,
        force_reprocess: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.force_reprocess = force_reprocess
        
        # Initialize components
        self.standardizer = ImageStandardizer(target_size=target_size)
        self.database = ImageDatabase(cache_dir=cache_dir)
        self.feature_extractor = FeatureExtractor(self.database)
        
        logger.info(f"Initialized pipeline with {max_workers} workers")
    
    def _get_image_files(self) -> List[Path]:
        """Get all image files from the images directory"""
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
        
        logger.info(f"Found {len(image_files)} images in {self.images_dir}")
        return sorted(image_files)
    
    def _standardize_image(self, image_path: Path) -> Tuple[str, bool]:
        """Standardize a single image"""
        image_id = image_path.stem
        
        # Skip if already processed and not forcing reprocess
        if not self.force_reprocess and self.database.has_image(image_id):
            logger.info(f"Skipping {image_id} (already processed)")
            return image_id, False
        
        try:
            # Standardize image
            img_array, metadata = self.standardizer.standardize(str(image_path))
            
            # Save to database
            self.database.save_image(image_id, img_array, metadata)
            
            logger.info(f"Standardized {image_id}: {metadata.original_size} -> {metadata.standardized_size}")
            return image_id, True
        
        except Exception as e:
            logger.error(f"Error standardizing {image_id}: {str(e)}")
            return image_id, False
    
    def standardize_all_images(self) -> List[str]:
        """Standardize all images in parallel"""
        logger.info("Starting image standardization...")
        image_files = self._get_image_files()
        
        processed_ids = []
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._standardize_image, img_path): img_path
                for img_path in image_files
            }
            
            for future in as_completed(futures):
                image_id, success = future.result()
                if success or self.database.has_image(image_id):
                    processed_ids.append(image_id)
        
        # Save metadata
        self.database.save_metadata()
        
        logger.info(f"Standardization complete: {len(processed_ids)} images processed")
        return processed_ids
    
    def _extract_features_for_image(self, image_id: str) -> ExtractedFeatures:
        """Extract features for a single image"""
        try:
            # Load standardized image
            img_array = self.database.load_image(image_id)
            
            if img_array is None:
                logger.error(f"Could not load image: {image_id}")
                return None
            
            # Extract features
            features = self.feature_extractor.extract_features(image_id, img_array)
            
            logger.info(f"Extracted features for {image_id} in {features.processing_time_total_ms:.2f}ms")
            return features
        
        except Exception as e:
            logger.error(f"Error extracting features for {image_id}: {str(e)}")
            return None
    
    def extract_all_features(self, image_ids: Optional[List[str]] = None) -> List[ExtractedFeatures]:
        """Extract features from all standardized images"""
        logger.info("Starting feature extraction...")
        
        # Use provided IDs or all images in database
        if image_ids is None:
            image_ids = list(self.database.metadata.keys())
        
        all_features = []
        
        # Process in parallel (using ThreadPoolExecutor for I/O bound operations)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._extract_features_for_image, image_id): image_id
                for image_id in image_ids
            }
            
            for future in as_completed(futures):
                features = future.result()
                if features is not None:
                    all_features.append(features)
        
        logger.info(f"Feature extraction complete: {len(all_features)} images processed")
        return all_features
    
    def save_features(self, features_list: List[ExtractedFeatures]):
        """Save extracted features to JSON file"""
        output_file = self.output_dir / "extracted_features.json"
        
        # Convert to serializable format
        features_data = []
        for features in features_list:
            data = {
                'image_id': features.image_id,
                'metadata': asdict(features.metadata),
                'processing_time_total_ms': features.processing_time_total_ms,
                'features': {
                    'attention_map': features.attention_map,
                    'characterization': features.characterization,
                    'emotional_tone': features.emotional_tone,
                    'text_branding': features.text_branding,
                    'visual_semantics': features.visual_semantics,
                    'llm_description': features.llm_description
                }
            }
            features_data.append(data)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(features_data, f, indent=2)
        
        logger.info(f"Saved features to {output_file}")
    
    def run(self) -> List[ExtractedFeatures]:
        """Run the complete extraction pipeline"""
        logger.info("=" * 60)
        logger.info("Starting Image Feature Extraction Pipeline")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Standardize all images
        logger.info("\n[Step 1/3] Standardizing images...")
        image_ids = self.standardize_all_images()
        
        # Step 2: Extract features
        logger.info("\n[Step 2/3] Extracting features...")
        features_list = self.extract_all_features(image_ids)
        
        # Step 3: Save results
        logger.info("\n[Step 3/3] Saving results...")
        self.save_features(features_list)
        
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info(f"Pipeline complete in {total_time:.2f} seconds")
        logger.info(f"Processed {len(features_list)} images")
        logger.info(f"Average time per image: {(total_time / len(features_list)):.2f}s")
        logger.info("=" * 60)
        
        return features_list


def main():
    """Main entry point for the extraction pipeline"""
    # Configure pipeline
    pipeline = ExtractionPipeline(
        images_dir="ads/images",
        cache_dir="processed_images",
        output_dir="extracted_features",
        target_size=(1024, 1024),  # Standard size for all images
        max_workers=4,  # Parallel processing workers
        force_reprocess=False  # Set to True to reprocess all images
    )
    
    # Run pipeline
    features = pipeline.run()
    
    return features


if __name__ == "__main__":
    main()
