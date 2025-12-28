"""
BERT4RecAlignMultiViewV2 - Re-export for RecBole model discovery

This file exists solely for RecBole's get_model() function which looks for
a file named after the model (bert4recalignmultiviewv2.py for BERT4RecAlignMultiViewV2).
The actual implementation is in bert4recalignmultiview.py.
"""

from recbole.model.sequential_recommender.bert4recalignmultiview import (
    BERT4RecAlignMultiViewV2,
    BERT4Rec_Align_MultiView_V2,
)

__all__ = [
    "BERT4RecAlignMultiViewV2",
    "BERT4Rec_Align_MultiView_V2",
]

