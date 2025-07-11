import numpy as np
from enum import Enum
from typing import Dict, List


class EventTypes(str, Enum):
    PRODUCT_BUY = "product_buy"
    SEARCH_QUERY = "search_query"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    PAGE_VISIT = "page_visit"


EVENT_TYPE_TO_COLUMNS: Dict[EventTypes, List[str]] = {
    EventTypes.PRODUCT_BUY: ["sku", "category", "price"],
    EventTypes.ADD_TO_CART: ["sku", "category", "price"],
    EventTypes.REMOVE_FROM_CART: ["sku", "category", "price"],
    EventTypes.PAGE_VISIT: ["url"],
    EventTypes.SEARCH_QUERY: [],
}
QUERY_COLUMN = "query"
EMBEDDINGS_DTYPE = np.float16
