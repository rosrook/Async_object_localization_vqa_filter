"""
Pipeline模块
"""
from .base_pipeline import BasePipeline
from .question_pipeline import QuestionPipeline
from .caption_pipeline import CaptionPipeline
from .objectcounting_pipeline import ObjectCountingPipeline
from .placerecognition_pipeline import PlaceRecognitionPipeline
from .textassociation_pipeline import TextAssociationPipeline
from .objectproportion_pipeline import ObjectProportionPipeline
from .objectorientation_pipeline import ObjectOrientationPipeline
from .objectabsence_pipeline import ObjectAbsencePipeline
from .objectposition_pipeline import ObjectPositionPipeline

__all__ = [
    "BasePipeline", 
    "QuestionPipeline", 
    "CaptionPipeline", 
    "PlaceRecognitionPipeline",
    "TextAssociationPipeline",
    "ObjectProportionPipeline",
    "ObjectPositionPipeline",
    "ObjectAbsencePipeline",
    "ObjectOrientationPipeline",
    "ObjectCountingPipeline"
    ]

