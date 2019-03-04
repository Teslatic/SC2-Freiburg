from pysc2.lib import actions
import numpy as np

SMART_ACTIONS_MOVE2BEACON = [
          actions.FUNCTIONS.no_op.id, # 0
          actions.FUNCTIONS.select_army.id, # 7
          actions.FUNCTIONS.Move_screen.id # 331
        ]

SMART_ACTIONS_DEFEAT_ROACHES = [
          actions.FUNCTIONS.no_op.id, # 0
          actions.FUNCTIONS.select_army.id, # 7
          actions.FUNCTIONS.Attack_screen.id, # 12
          actions.FUNCTIONS.Move_screen.id # 331
        ]

SMART_ACTIONS_GRID = [
          actions.FUNCTIONS.no_op.id, # 0
          actions.FUNCTIONS.select_army.id, # 7
          actions.FUNCTIONS.Move_screen.id # 331
        ]

SMART_ACTIONS_COMPASS = [
          'left',
          'up',
          'right',
          'down',
          actions.FUNCTIONS.select_army.id # 7
        ]

# Action sets as documented in reaver:
# https://github.com/inoryy/reaver/blob/master/reaver/envs/sc2.py
# sensible action set for all minigames
SMART_ACTIONS_PYSC2 = [
			actions.FUNCTIONS.no_op.id, # 0
			actions.FUNCTIONS.move_camera.id, # 1
			actions.FUNCTIONS.select_point.id, # 2
			actions.FUNCTIONS.select_rect.id, # 3
			actions.FUNCTIONS.select_control_group.id, # 4
			actions.FUNCTIONS.select_idle_worker.id, # 6
			actions.FUNCTIONS.select_army.id, # 7
			# 11
			actions.FUNCTIONS.Attack_screen.id, # 12
			actions.FUNCTIONS.Attack_minimap.id, # 13

			actions.FUNCTIONS.Build_Barracks_screen.id, # 42
			actions.FUNCTIONS.Build_CommandCenter_screen.id, # 44
			actions.FUNCTIONS.Build_EngineeringBay_screen.id, # 50
            # 71,
			# 72,
			# 73,
			# 74,
			# 79,
			actions.FUNCTIONS.Build_SupplyDepot_screen.id, # 91
			# 140
			# 168
			actions.FUNCTIONS.Effect_CalldownMULE_screen.id, # 183
			actions.FUNCTIONS.Effect_Stim_quick.id, # 234
			# 239
			# 261
			# 264
			# 269
			# 274
			actions.FUNCTIONS.Morph_OrbitalCommand_quick.id, # 309
			# 318
			actions.FUNCTIONS.Move_screen.id, # 331
			actions.FUNCTIONS.Move_minimap.id, # 332
			actions.FUNCTIONS.Patrol_screen.id, # 333
			actions.FUNCTIONS.Patrol_minimap.id, # 334
			# 335
			# 336
			actions.FUNCTIONS.Smart_screen.id, # 451
			actions.FUNCTIONS.Smart_minimap.id, # 452
			# 453
			# 477
			actions.FUNCTIONS.Train_SCV_quick.id, # 490
			]
ALL_ACTIONS_PYSC2_IDX = [0,1,2,3,4,6,7,11,12,13,42,44,50, 71, 72, 73,74,79,91,140,168,183, 234, 239, 261,264, 269,274, 309,318, 331,332,333,334,335,336,451, 452, 453, 477, 490]
# Q_IDX = range(len(ALL_ACTIONS_PYSC2_IDX))
# PYSC2_to_Qvalueindex = zip(ALL_ACTIONS_PYSC2_IDX, Q_IDX)

PYSC2_to_Qvalueindex = {
                        0:0,
                        1:1,
                        2:2,
                        3:3,
                        4:4,
                        6:5,
                        7:6,
                        11:7,
                        12:8,
                        13:9,
                        42:10,
                        44:11,
                        50:12,
                        71:13,
                        72:14,
                        73:15,
                        74:16,
                        79:17,
                        91:18,
                        140:19,
                        168:20,
                        183:21,
                        234:22,
                        239:23,
                        261:24,
                        264:25,
                        269:26,
                        274:27,
                        309:28,
                        318:29,
                        331:30,
                        332:31,
                        333:32,
                        334:33,
                        335:34,
                        336:35,
                        451:36,
                        452:37,
                        453:38,
                        477:39,
                        490:40
}

ALL_ACTIONS_PYSC2 = [
			actions.FUNCTIONS.no_op, # 0
			actions.FUNCTIONS.move_camera, # 1
			actions.FUNCTIONS.select_point, # 2
			actions.FUNCTIONS.select_rect, # 3
			actions.FUNCTIONS.select_control_group, # 4
			actions.FUNCTIONS.select_idle_worker, # 6
			actions.FUNCTIONS.select_army, # 7
			actions.FUNCTIONS.build_queue, # 11
			actions.FUNCTIONS.Attack_screen, # 12
			actions.FUNCTIONS.Attack_minimap, # 13
			actions.FUNCTIONS.Build_Barracks_screen, # 42
			actions.FUNCTIONS.Build_CommandCenter_screen, # 44
			actions.FUNCTIONS.Build_EngineeringBay_screen, # 50

			actions.FUNCTIONS.Build_Reactor_quick, # 71
			actions.FUNCTIONS.Build_Reactor_screen, # 72
			actions.FUNCTIONS.Build_Reactor_Barracks_quick, # 73
			actions.FUNCTIONS.Build_Reactor_Barracks_screen, # 74
			actions.FUNCTIONS.Build_Refinery_screen, # 79
			actions.FUNCTIONS.Build_SupplyDepot_screen, # 91


			actions.FUNCTIONS.Cancel_quick, # 140
			actions.FUNCTIONS.Cancel_Last_quick, # 168
			actions.FUNCTIONS.Effect_CalldownMULE_screen, # 183
			actions.FUNCTIONS.Effect_Stim_quick, # 234
			actions.FUNCTIONS.Effect_SupplyDrop_screen, # 239
			actions.FUNCTIONS.Halt_quick, # 261
			actions.FUNCTIONS.Harvest_Gather_screen, # 264
			actions.FUNCTIONS.Harvest_Return_quick, # 269
			actions.FUNCTIONS.HoldPosition_quick, # 274

			actions.FUNCTIONS.Morph_OrbitalCommand_quick, # 309
			actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick, # 318
			actions.FUNCTIONS.Move_screen, # 331
			actions.FUNCTIONS.Move_minimap, # 332
			actions.FUNCTIONS.Patrol_screen, # 333
			actions.FUNCTIONS.Patrol_minimap, # 334
			actions.FUNCTIONS.Rally_Units_screen, # 335
			actions.FUNCTIONS.Rally_Units_minimap, # 336

			actions.FUNCTIONS.Smart_screen, # 451
			actions.FUNCTIONS.Smart_minimap, # 452
			actions.FUNCTIONS.Stop_quick, # 453
			actions.FUNCTIONS.Train_Marine_quick, # 477
			actions.FUNCTIONS.Train_SCV_quick, # 490
			]

# full action space, including outdated / unusable to current race / usable only in certain cases
action_ids = [f.id for f in actions.FUNCTIONS]
