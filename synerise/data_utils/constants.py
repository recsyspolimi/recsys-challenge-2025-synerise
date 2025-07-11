from enum import Enum


class EventTypes(str, Enum):
    PRODUCT_BUY = "product_buy"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    PAGE_VISIT = "page_visit"
    SEARCH_QUERY = "search_query"


PROPERTIES_FILE = "product_properties.parquet"

DAYS_IN_TARGET = 14
