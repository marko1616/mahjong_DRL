
from random import shuffle
from random import seed as random_seed

from mahjong.shanten import Shanten
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.hand_calculating.hand_config import OptionalRules
from mahjong.meld import Meld

from typing import Optional

from rich import print

from copy import deepcopy

RESET_SIGN = "RESET"
STOP_SIGN = "STOP"

"""
## 动作空间定义
    ### 动作:
        - 打牌:
            + 1-9m 1-9p 1-9s 1-7z(东南西北白发中) 1-34
        - 吃:
            + 上吃:eg.23吃1 35
            + 中吃:eg.13吃2 36
            + 下吃:eg.12吃3 37
        - 碰:
            + AA碰A 38
        - 杠:
            + 明杠:AAA明杠他人A 39
            + 加杠:自己碰的AAA加杠自己摸入的A 40
            + 暗杠:自己摸入AAAA可选的杠 41
        - 拔北:
            + 拔北:三人限定 42
        - 立直:
            + 门清立 43
        - 荣和:
            + 和别家的牌 44
        - 自摸:
            + 摸入和牌 45
        - 跳过
            + 如果不吃碰杠和自摸立直 46
    ### 备注
        - 备注:吃碰明杠无需考虑吃谁的因为tokens列表的t-1一定是上家
        - 备注:吃碰等需要再打出一张牌的操作按照打牌计算
        - 备注:词表大小为:184
"""


class MahjongEnv:
    REWARD_WEIGHT_SHANTEN: float = 30.
    PENALTY_AVA_NUM: float = 1.2
    SCORE_WEIGHT: float = 0.1
    REWARD_RICHI: float = 6.
    REWARD_NO_YAKU: float = -100.
    REWARD_OPEN_TANYAO: float = -10
    NAME_MAP = {"m": "man", "p": "pin", "s": "sou", "z": "honors"}

    def __init__(self, is_three: bool = False):
        self.shanten_calculator = Shanten()
        self.score_calculator = HandCalculator()
        # self.reset(is_three=is_three)

    def reset(self, is_three: bool = False):
        # [TODO] 增加下面两个宝牌划分和上面分牌划分的三人支持
        # 重置游戏环境。如果is_three为True，则游戏为三人麻将，否则为四人麻将。
        # 初始化剩余牌堆，包含9种每种4张的“万、饼、索”（man, pin, sou），以及7种每种4张的字牌（1-7z）
        self.done: bool = False
        self.pending_done: bool = False  # 多人同时胡牌的等待flag
        self.seat_now: int = 0  # 0-3
        self.claiming_from = 0

        # dict[str_type:[[int_player_index,action_index,list[Any]],构建列表是为了优先级。list[any]为上一轮判定可以鸣牌的必要信息
        # 必要信息:对自摸和荣和是和的牌用于点数计算
        # 对吃碰杠是吃碰杠了什么牌str
        self.pending_claiming: dict[str, list[int, str, Optional[list[any]]]] = {
            "tsumo": [], "ron": [], "richi": [],"pei": [], "kang": [], "pong": [], "chi": []}

        self.remain_tiles = [f"{i + 1}{j}" for i in range(9) for j in "mps"] * 4 + [
            f"{i + 1}z" for i in range(7)] * 4
        shuffle(self.remain_tiles)  # 打乱剩余牌堆的顺序

        # 初始化历史动作记录
        self.history_tokens = []
        # 初始化手牌，为每个花色创建一个包含数值1-9的字典，字牌为1-7
        self.hand_tiles = {
            j: {i+1: 0 for i in range(9)} for j in ["man", "pin", "sou"]}
        self.hand_tiles.update({"honors": {i+1: 0 for i in range(7)}})

        # 创建鸣牌记录
        self.melds = [[] for _ in range(3)] if is_three else [[] for _ in range(4)]
        self.kang_count = 0  # 如果四次杠要流局除非能和

        self.status = {"richi": False,
                       "shocking": False, "has_open_tanyao": False, "last_available_num": None, "last_shanten": 6, "last_get_tile": "", "first_round": True}
        # 根据游戏是三人还是四人，复制初始化的手牌结构
        self.hand_tiles = [deepcopy(self.hand_tiles) for _ in range(3)] if is_three else [
            deepcopy(self.hand_tiles) for _ in range(4)]
        self.status = [deepcopy(self.status) for _ in range(3)] if is_three else [
            deepcopy(self.status) for _ in range(4)]

        # 为每个玩家发13张牌
        for player_index in range(4):
            random_tiles = self.remain_tiles[:13]  # 取前13张牌
            self.remain_tiles = self.remain_tiles[13:]  # 更新剩余牌堆

            # 将这13张牌分配到相应玩家的手牌中
            for tile in random_tiles:
                self._delta_bucket(self.hand_tiles[player_index], tile)

        # 设置宝牌指示牌，取剩余牌的前10张
        self.dora_list = self.remain_tiles[:10]
        self.remain_tiles = self.remain_tiles[10:]

        # 设置岭上牌，取接下来的4张
        self.rinshan = self.remain_tiles[:4]
        self.remain_tiles = self.remain_tiles[4:]

        return self.step(0, start=True)

    def _str_to_34(self, tile_str: str) -> list:
        temp_list = [0]*34
        temp_list[list(self.NAME_MAP.keys()).index(tile_str[-1])*9+int(tile_str[0])-1] = 1
        return temp_list

    def _delta_bucket(self, bucket: dict[str, dict[str, int]], target: str, gain: int = 1) -> None:
        assert bucket[self.NAME_MAP[target[-1]]][int(target[0])] + gain >= 0
        bucket[self.NAME_MAP[target[-1]]][int(target[0])] += gain

    def _get_first_claiming_event(self, pop: bool = False) -> list[int, int] | bool:
        for claiming_type in self.pending_claiming:
            if self.pending_claiming[claiming_type]:
                if pop:
                    return self.pending_claiming[claiming_type].pop(0)
                else:
                    return self.pending_claiming[claiming_type][0]
        return False

    def tile_34_to_str(card: int) -> str:
        index_to_card_map = {0: "m", 1: "p", 2: "s"}
        if card < 28:  # 非字牌
            return f"{card-int((card-1)/9)*9}{index_to_card_map[int((card-1)/9)]}"
        else:
            return f"{card-27}z"

    def _34_to_str(self, card: int) -> str:
        index_to_card_map = {0: "m", 1: "p", 2: "s"}
        if card < 28:  # 非字牌
            return f"{card-int((card-1)/9)*9}{index_to_card_map[int((card-1)/9)]}"
        else:
            return f"{card-27}z"

    def _get_hand_34_array(self, player_index: int) -> list[int]:
        params_str = {"man": "", "pin": "", "sou": "", "honors": ""}
        for type_key in self.hand_tiles[player_index]:
            temp_str = ""
            for index_key in self.hand_tiles[player_index][type_key]:
                temp_str += f"{index_key}" * \
                    self.hand_tiles[player_index][type_key][index_key]
            params_str[type_key] = temp_str

        return TilesConverter.string_to_34_array(**params_str)

    def _add_action_list(self, actions: list, add_action: list) -> list:
        return [x or y for x, y in zip(actions, add_action)]

    def _handle_meld(self, action, claiming_event) -> None:
        # 移除别的鸣牌
        reward = 0
        for key in {"kang", "pong", "chi"}:
            self.pending_claiming[key] = []
        # 杠
        if action in {41, 40, 39}:
            self.kang_count += 1
            temp_tiles = [0]*34
            temp_tiles[claiming_event[2]] += 4
            temp_tiles = TilesConverter.to_136_array(temp_tiles)
            if action == 41:
                self.melds[self.seat_now].append(Meld(meld_type=Meld.KAN,tiles=temp_tiles,opened=False))
                self._delta_bucket(
                    self.hand_tiles[self.seat_now], self._34_to_str(claiming_event[2]+1), gain=-4)
            elif action == 40:
                self.melds[self.seat_now].append(Meld(meld_type=Meld.KAN,tiles=temp_tiles))
                temp_tiles = [0]*34
                temp_tiles[claiming_event[2]] += 3
                temp_tiles = TilesConverter.to_136_array(temp_tiles)
                for index, meld in enumerate(self.melds[self.seat_now]):
                    if str(meld) == str(Meld(meld_type=Meld.PON,tiles=temp_tiles)):
                        break
                del self.melds[self.seat_now][index]
                self._delta_bucket(
                    self.hand_tiles[self.seat_now], self._34_to_str(claiming_event[2]+1), gain=-1)
            else:
                self.melds[self.seat_now].append(Meld(meld_type=Meld.KAN,tiles=temp_tiles))
                self.status[self.seat_now]["has_open_tanyao"] = True
                reward += self.REWARD_OPEN_TANYAO
                self._delta_bucket(
                    self.hand_tiles[self.seat_now], self._34_to_str(claiming_event[2]+1), gain=-3)
            # 发岭上
            self._delta_bucket(
                self.hand_tiles[self.seat_now], self.rinshan.pop(0), gain=1)

        # 碰
        elif action == 38:
            temp_tiles = [0]*34
            temp_tiles[claiming_event[2]] += 3
            temp_tiles = TilesConverter.to_136_array(temp_tiles)
            self.melds[self.seat_now].append(Meld(meld_type=Meld.PON,tiles=temp_tiles))
            self.status[self.seat_now]["has_open_tanyao"] = True
            reward += self.REWARD_OPEN_TANYAO
            self._delta_bucket(
                self.hand_tiles[self.seat_now], self._34_to_str(claiming_event[2]+1), gain=-2)

        # 吃
        else:
            self.status[self.seat_now]["has_open_tanyao"] = True
            reward += self.REWARD_OPEN_TANYAO
            if action == 37:
                temp_tiles = [0]*34
                temp_tiles[claiming_event[2]] += 1
                temp_tiles[claiming_event[2]-1] += 1
                temp_tiles[claiming_event[2]-2] += 1
                temp_tiles = TilesConverter.to_136_array(temp_tiles)
                self.melds[self.seat_now].append(Meld(meld_type=Meld.CHI,tiles=temp_tiles))
                self._delta_bucket(
                    self.hand_tiles[self.seat_now], self._34_to_str(claiming_event[2]), gain=-1)
                self._delta_bucket(
                    self.hand_tiles[self.seat_now], self._34_to_str(claiming_event[2]-1), gain=-1)
            elif action == 36:
                temp_tiles = [0]*34
                temp_tiles[claiming_event[2]] += 1
                temp_tiles[claiming_event[2]-1] += 1
                temp_tiles[claiming_event[2]+1] += 1
                temp_tiles = TilesConverter.to_136_array(temp_tiles)
                self.melds[self.seat_now].append(Meld(meld_type=Meld.CHI,tiles=temp_tiles))
                self._delta_bucket(
                    self.hand_tiles[self.seat_now], self._34_to_str(claiming_event[2]+2), gain=-1)
                self._delta_bucket(
                    self.hand_tiles[self.seat_now], self._34_to_str(claiming_event[2]), gain=-1)
            else:
                temp_tiles = [0]*34
                temp_tiles[claiming_event[2]] += 1
                temp_tiles[claiming_event[2]+1] += 1
                temp_tiles[claiming_event[2]+2] += 1
                temp_tiles = TilesConverter.to_136_array(temp_tiles)
                self.melds[self.seat_now].append(Meld(meld_type=Meld.CHI,tiles=temp_tiles))
                self._delta_bucket(
                    self.hand_tiles[self.seat_now], self._34_to_str(claiming_event[2]+3), gain=-1)
                self._delta_bucket(
                    self.hand_tiles[self.seat_now], self._34_to_str(claiming_event[2]+2), gain=-1)
        return reward

    def _get_discard_mask(self, player_index) -> list[int]:
        if self.status[player_index]["richi"]:
            return [1 if i else 0 for i in self._str_to_34(self.status[player_index]["last_get_tile"])] + [0]*12
        else:
            return [1 if i else 0 for i in self._get_hand_34_array(player_index)] + [0]*12

    def _hai_index_to_34(self, hai_index) -> list[int]:
        list_34 = [0]*34
        list_34[hai_index-1] = 1
        return list_34

    def _get_dora(self,end:bool=False,richi:bool=False,return_34:bool=False) -> list[int]:
        dora_str = []
        for i in range(self.kang_count+1):
            dora_str.append(self.dora_list[:5][i])
        if end and richi:
            for i in range(self.kang_count+1):
                dora_str.append(self.dora_list[5:][i])
        dora_list = [0]*34
        for dora in dora_str:
            dora_list[self._str_to_34(dora).index(1)] += 1
        if return_34:
            return dora_list
        return TilesConverter.to_136_array(dora_list)

    def _get_available_num(self, player_index: int) -> int:
        tiles = self._get_hand_34_array(player_index)
        now_shanten = self.shanten_calculator.calculate_shanten(tiles)
        available_tiles = []
        for index,num in enumerate(tiles):
            if num > 0:
                tiles_deleted = tiles.copy()
                tiles_deleted[index] -=1
                remain_tile_type = set(self.remain_tiles)
                for get_tile in remain_tile_type:
                    tiles_to_calcu = tiles_deleted.copy()
                    tiles_to_calcu[self._str_to_34(get_tile).index(1)] += 1
                    if self.shanten_calculator.calculate_shanten(tiles_to_calcu) < now_shanten:
                        available_tiles.append(get_tile)
        available_num = 0
        for item in set(available_tiles):
            available_num += self.remain_tiles.count(item)
        return available_num

    def step(self, action: int, start: bool = False) -> tuple[dict, int, bool, list]:
        if not start:
            self.history_tokens.append(action)

        action = action - 46*self.seat_now
        reward = 0
        # 用于追踪摸牌导致向听数奖励并添加到历史R_t-1的info
        reward_update = 0
        claiming = False
        action_mask = [0]*46

        # 鸣牌&特殊动作处理
        claiming_event = self._get_first_claiming_event()
        if claiming_event and not start:
            if action == 46:
                # 跳过
                self._get_first_claiming_event(pop=True)
            elif action == 45:
                # 自摸,事件附加信息是自摸前的13张牌和自摸牌
                self.done = True
                self._get_first_claiming_event(pop=True)

                reward = claiming_event[2].cost["main"]*self.SCORE_WEIGHT
            elif action == 44:
                # 荣和,事件附加信息是和前的13张牌和和的牌
                self._get_first_claiming_event(pop=True)
                self.pending_done = True
    
                reward = claiming_event[2].cost["main"]*self.SCORE_WEIGHT

            elif action == 43:
                # 立直
                self._get_first_claiming_event(pop=True)
                self.status[self.seat_now]["richi"] = True

                reward = self.REWARD_RICHI

            elif action == 42:
                # [TODO]三人拔北
                pass

            elif action in {35, 36, 37, 38, 39, 40, 41}:
                # 处理吃、碰、各类杠
                self._get_first_claiming_event(pop=True)
                reward += self._handle_meld(action, claiming_event)

                claiming = True

        # 处理打牌
        if action < 35 and not start:
            # 计算本次摸入的牌有无导致向听数减小
            shanten_change = False
            shanten = self.shanten_calculator.calculate_shanten(self._get_hand_34_array(self.seat_now))
            if not self.status[self.seat_now]["first_round"] and shanten != self.status[self.seat_now]["last_shanten"] and self.status[self.seat_now]["last_shanten"]:
                    reward_update = (reward_update + self.status[self.seat_now]["last_shanten"] - shanten) if shanten <= self.status[self.seat_now]["last_shanten"] else (reward_update + (self.status[self.seat_now]["last_shanten"] - shanten)*2)
                    reward_update *= self.REWARD_WEIGHT_SHANTEN
                    shanten_change = True
            self.status[self.seat_now]["last_shanten"] = shanten
            self.status[self.seat_now]["first_round"] = False

            available_num = self._get_available_num(self.seat_now)

            if not self.status[self.seat_now]["first_round"] and available_num != self.status[self.seat_now]["last_available_num"] and not shanten_change and self.status[self.seat_now]["last_available_num"]:
                    # 向听数变化时不计入有效进牌
                    reward_update = (reward_update + self.status[self.seat_now]["last_available_num"] - available_num) if available_num <= self.status[self.seat_now]["last_available_num"] else (reward_update + (self.status[self.seat_now]["last_available_num"] - available_num)*self.PENALTY_AVA_NUM)
            self.status[self.seat_now]["last_available_num"] = available_num
            self.status[self.seat_now]["first_round"] = False
            
            self.claiming_from = self.seat_now
            self._delta_bucket(self.hand_tiles[self.seat_now], self._34_to_str(action), gain=-1)
            # 添加鸣牌列表
            for player_index in range(3):
                if player_index == self.seat_now:
                    continue
                temp_tiles = self._get_hand_34_array(player_index)
                temp_tiles[action-1] += 1
                if self.shanten_calculator.calculate_shanten(temp_tiles) == -1:
                    # 添加荣和
                    copied_bucket = deepcopy(self.hand_tiles[player_index])
                    self._delta_bucket(copied_bucket,target=self._34_to_str(action))
                    result = self.score_calculator.estimate_hand_value(TilesConverter.to_136_array(temp_tiles),
                                                                TilesConverter.to_136_array(self._hai_index_to_34(action))[0],
                                                                melds=self.melds[player_index],
                                                                dora_indicators=self._get_dora(end=True,richi=self.status[self.seat_now]["richi"]),
                                                                config=HandConfig(is_riichi=self.status[self.seat_now]["richi"],
                                                                                    options=OptionalRules(self.status[self.seat_now]["has_open_tanyao"])))
                    if not result.error:
                        # 不存在役等乱七八糟的error
                        self.pending_claiming["ron"].append([player_index, 44, result])
                    elif result.error == self.score_calculator.ERR_NO_YAKU:
                        reward += self.REWARD_NO_YAKU
                if not self.status[player_index]["richi"]:
                    if action <= 27 and (player_index - 1 == self.seat_now or (player_index == 0 and self.seat_now == 3)):# 不能吃字牌和只能吃上家
                        # 上吃
                        if action+2 <= 27:
                            if (self._get_hand_34_array(player_index)[action-1+1] > 0 and self._get_hand_34_array(player_index)[action-1+2] > 0 and
                                int((action) / 9) == int((action+1) / 9) == int((action+2) / 9)
                                ):
                                self.pending_claiming["chi"].append(
                                    [player_index, 35, action-1])
                        # 中吃
                        if action+1 <= 27 and action-1 > 0:
                            if (self._get_hand_34_array(player_index)[action-1-1] > 0 and self._get_hand_34_array(player_index)[action-1+1] > 0 and
                                int((action) / 9) == int((action-1) / 9) == int((action+1) / 9) and
                                action - 1 > 0
                                ):
                                self.pending_claiming["chi"].append(
                                    [player_index, 36, action-1])
                        # 下吃
                        if action-2 > 0:
                            if (self._get_hand_34_array(player_index)[action-1-1] > 0 and self._get_hand_34_array(player_index)[action-1-2] > 0 and
                                int((action) / 9) == int((action-1) / 9) == int((action-2) / 9) and
                                action - 1 > 0 and
                                action - 2 > 0
                                ):
                                self.pending_claiming["chi"].append(
                                    [player_index, 37, action-1])
                    # 碰
                    if (self._get_hand_34_array(player_index)[action-1] >= 2):
                        self.pending_claiming["pong"].append(
                            [player_index, 38, action-1])
                    # 明杠
                    if (self._get_hand_34_array(player_index)[action-1] == 3):
                        self.pending_claiming["pong"].append(
                            [player_index, 39, action-1])

        # 开始对下回合进行计算
        last_claiming_event = claiming_event
        claiming_event = self._get_first_claiming_event()
        if last_claiming_event and not claiming_event:
            # 如果所有人跳过则恢复正常顺序
            self.seat_now = self.claiming_from

        if (self.pending_done and not self.pending_claiming["ron"]) or (not self.remain_tiles) or (self.kang_count == 4 and not (self.pending_claiming["tsumo"] or self.pending_claiming["ron"])):
            self.done = True
        elif claiming_event:
            # 如果存在等待的鸣牌则继续
            self.seat_now = claiming_event[0]
            temp_list = [0]*46
            # 鸣牌或跳过
            temp_list[claiming_event[1]-1] = 1
            temp_list[45] = 1
            action_mask = self._add_action_list(action_mask, temp_list)
        elif claiming:
            # 有人鸣牌则下一轮给他打出
            self.seat_now = last_claiming_event[0]
            action_mask = self._add_action_list(
                action_mask, self._get_discard_mask(self.seat_now))
        elif last_claiming_event and last_claiming_event[1] in {45, 43, 42, 41, 40}:
            # 如果是立直，自摸，暗杠，加杠，拔北的跳过则也需自己打出
            # print("DROP")
            self.seat_now = last_claiming_event[0]
            action_mask = self._add_action_list(action_mask, self._get_discard_mask(self.seat_now))
        elif not self.remain_tiles: 
            self.done = True
        else:
            # 发牌
            if not start:
                self.seat_now += 1
            if self.seat_now == 4:
                self.seat_now = 0

            get_tile = self.remain_tiles.pop(0)
            self._delta_bucket(
                self.hand_tiles[self.seat_now], get_tile)
            self.status[self.seat_now]["last_get_tile"] = get_tile
            # print(self.hand_tiles[self.seat_now])
            # print(self._get_hand_34_array(self.seat_now))
            shanten = self.shanten_calculator.calculate_shanten(self._get_hand_34_array(self.seat_now))
            if shanten == -1:
                # 如果自摸
                temp_list = [0]*46
                temp_list[44]=1
                temp_list[45]=1
                result = self.score_calculator.estimate_hand_value(TilesConverter.to_136_array(self._get_hand_34_array(self.seat_now)),
                                                TilesConverter.to_136_array(self._str_to_34(get_tile))[0],
                                                melds=self.melds[self.seat_now],
                                                dora_indicators=self._get_dora(end=True,richi=self.status[self.seat_now]["richi"]),
                                                config=HandConfig(is_riichi=self.status[self.seat_now]["richi"],
                                                                  options=OptionalRules(self.status[self.seat_now]["has_open_tanyao"])))
                if not result.error:
                    self.pending_claiming["tsumo"].append([self.seat_now, 45, result])
                    action_mask = self._add_action_list(action_mask, temp_list)
                else:
                    action_mask = self._add_action_list(action_mask, self._get_discard_mask(self.seat_now))
                    if result.error == self.score_calculator.ERR_NO_YAKU:
                        reward += self.REWARD_NO_YAKU
            elif shanten == 0 and not self.melds[self.seat_now] and not self.status[self.seat_now]["richi"]:
                # 如果可以立直
                temp_list = [0]*46
                temp_list[42]=1
                temp_list[45]=1
                self.pending_claiming["richi"].append([self.seat_now, 43, [self._get_hand_34_array(self.seat_now), get_tile]])
                action_mask = self._add_action_list(action_mask, temp_list)
            else:
                action_mask = self._add_action_list(action_mask, self._get_discard_mask(self.seat_now))

        # 添加偏移
        action_mask = [0] + [0]*46*self.seat_now + \
            action_mask + [0]*46*(3-self.seat_now)

        return {
            "tokens": self.history_tokens,
            "is_three": 0,
            "seat": self.seat_now,
            "hand": self._get_hand_34_array(self.seat_now),
        }, reward, self.done, {
            "action_mask": action_mask,
            "reward_update": reward_update
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def env_process(queue_in, queue_out):
    env = MahjongEnv()
    while True:
        action = queue_in.get()
        if action == STOP_SIGN:
            break

        if action == RESET_SIGN:
            queue_out.put(env.reset())
        else:
            queue_out.put(env.step(action))

def action_to_str(action:int):
    seat = int((action-1) / 46)
    action = ((action-1) % 46)+1
    if action in range(1,35):
        return f"{seat}号玩家打出{MahjongEnv.tile_34_to_str(action)}"
    elif action in range(35,38):
        return f"{seat}号玩家吃牌"
    elif action == 38:
        return f"{seat}号玩家碰牌"
    elif action in range(39, 41):
        return f"{seat}号玩家杠牌"
    elif action == 42:
        return f"{seat}号玩家拔北"
    elif action == 43:
        return f"{seat}号玩家立直"
    elif action == 44:
        return f"{seat}号玩家荣和"
    elif action == 45:
        return f"{seat}号玩家自摸"
    elif action == 46:
        return f"{seat}号玩家跳过"
    else:
        # 不应该执行这个
        assert False

if __name__ == "__main__":
    # 测试
    import random
    import time
    from collections import deque

    def random_one_index(multihot):
        # 找出所有值为1的索引
        zero_indices = [i for i, value in enumerate(multihot) if value == 1]
        # 随机选择一个索引
        if zero_indices:
            return random.choice(zero_indices)
        else:
            raise BaseException("actions mask error")

    seed = 0
    time_list = deque(maxlen=8000)
    random_seed(seed)
    obj = MahjongEnv()
    status, reward, done, info = obj.reset()
    while True:
        time_s = time.time()
        action = random_one_index(info["action_mask"])
        status, reward, done, info = obj.step(action)
        time_list.append(time.time()-time_s)
        # print(status["actions"])
        print(action_to_str(action),end=" ")
        if done:
            print(f"DONE #{seed}\nStep Time: {1/(sum(time_list)/len(time_list))} steps/second")
            status, reward, done, info = obj.reset()
            seed += 1
            random_seed(seed)
        for player in range(3):
            res = obj._get_hand_34_array(player)
            b = 0
            for i in res:
                b += i
            if not b:
                print(f"index:{player}")
                print(f"{obj.melds}")
                breakpoint()