from typing import TypedDict


class KeyMapping(TypedDict, total=False):
    escape: int
    backspace: int
    space: int
    enter: int

    left_square_bracket: int  # [
    right_square_bracket: int  # ]

    left: int
    right: int
    up: int
    down: int

    plus: int  # +
    minus: int  # -
    equal: int  # =


#
COMMON_KEYMAPPING: KeyMapping = {
    'escape': 27,
    'space': 32,
    'enter': 13,

    'left_square_bracket': 91,
    'right_square_bracket': 93,

    # ord
    'plus': ord('+'),
    'minus': ord('-'),
    'equal': ord('=')
}

WIN_KEYMAPPING: KeyMapping = {
    **COMMON_KEYMAPPING,
    'backspace': 8,
    'left': 2424832,
    'right': 2555904,
    'up': 2490368,
    'down': 2621440
}

MAC_KEYMAPPING: KeyMapping = {
    **COMMON_KEYMAPPING,
    'backspace': 127,
    'left': 2,
    'right': 3,
    'up': 0,
    'down': 1

}

LINUX_KEYMAPPING: KeyMapping = {
    **COMMON_KEYMAPPING,
    'backspace': 8,
    'left': 81,
    'right': 83,
    'up': 82,
    'down': 84
}
