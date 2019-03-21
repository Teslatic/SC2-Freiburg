from pysc2.lib import actions

SMART_ACTIONS_MOVE2BEACON = [
          actions.FUNCTIONS.select_army.id,
          actions.FUNCTIONS.Move_screen.id
        ]

SMART_ACTIONS_MOVE2BEACON_SIMPLE = [
          actions.FUNCTIONS.Move_screen.id,
          actions.FUNCTIONS.no_op.id
        ]

SMART_ACTIONS_COMPASS = [
          'left',
          'up',
          'right',
          'down'
        ]
