from enum import Enum


class DrawerClass(str, Enum):
    DRAWER = "Drawer"
    CABINET = "Cabinet"
    CHEST = "Chest"
    NIGHT_STAND = "NightStand"
    SIDE_TABLE = "SideTable"
    BUFFET = "Buffet"
    CEDAR_CHEST = "CedarChest"
    CHINA_CABINET = "ChinaCabinet"
    CREDENZA = "Credenza"
    CUPBOARD = "Cupboard"
    AIRING_CUPBOARD = "AiringCupboard"
    HOPE_CHEST = "HopeChest"
    HUTCH = "Hutch"
    LOCKER = "Locker"
    FOOTLOCKER = "Footlocker"
    MEDICINE_CHEST = "MedicineChest"
    PANTRY = "Pantry"
    SIDEBOARD = "Sideboard"
    WARDROBE = "Wardrobe"
    CABINETWORK = "Cabinetwork"
    DISHWASHER = "Dishwasher"
    REFRIGERATOR = "Refrigerator"
    # Not included since they are a set of drawers:
    # ARMOIRE = "Armoire"
    # DRESSER = "Dresser"
    # FILING_CABINET = "FilingCabinet"
    # CHEST_OF_DRAWERS = "ChestOfDrawers"


class HandleClass(str, Enum):
    HANDLE = "Handle"
    KNOB = "Knob"
    DOORKNOB = "Doorknob"
    PULL = "Pull"
    BELLPULL = "Bellpull"
    PULL_CHAIN = "PullChain"


DRAWER_CLASSES = set(DrawerClass)
HANDLE_CLASSES = set(HandleClass)
