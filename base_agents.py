# -*- coding: utf-8 -*-
import math
import statistics
from typing import Dict
import negmas
import scml

from collections import defaultdict
import random
from negmas import ResponseType
from scml.oneshot import *
from scml.utils import anac2024_oneshot

#tournament
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#BaseAgent
class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs
            else ResponseType.REJECT_OFFER
        )

    def best_offer(self, negotiator_id):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]

        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = self._find_good_price(ami)
        return tuple(offer)

    def _find_good_price(self, ami):
        """Finds a good-enough price."""
        unit_price_issue = ami.issues[UNIT_PRICE]
        if self._is_selling(ami):
            return unit_price_issue.max_value
        return unit_price_issue.min_value

    def is_seller(self, negotiator_id):
        return negotiator_id in self.awi.current_negotiation_details["sell"].keys()

    def _needed(self, negotiator_id=None):
        return (
            self.awi.needed_sales
            if self.is_seller(negotiator_id)
            else self.awi.needed_supplies
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

class FNAgent(SimpleAgent):
    def before_step(self):
        super().before_step()
        if self.awi.level == 0:
            self.q_need = self.awi.current_exogenous_input_quantity
        elif self.awi.level == 1:
            self.q_need = self.awi.current_exogenous_output_quantity
        self.q_opp_offer = defaultdict(lambda: float("inf"))

    def propose(self, negotiator_id: str, state) -> "Outcome":
        if self.q_need <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        offer = super().propose(negotiator_id, state)
        if offer is None:
            return None
        if self.awi.current_step/self.awi.n_steps >= 0.7:
            return offer
        offer = list(offer)

        quantity_issue = ami.issues[QUANTITY]

        if self._is_selling(ami):
            #print('売る'+str(self.q_need))
            opponent = ami.annotation["buyer"]
            resource = self.q_need
            if len(self.q_opp_offer.keys()) >= 1: #mの設定
                fake_needs = int(resource * 1.5) #売りたい数をかさましする
                offer[QUANTITY] = self.best_q(opponent, fake_needs)
                #offer[QUANTITY] = fake_needs / len(self.q_opp_offer.keys())
                if offer[QUANTITY] >= quantity_issue.max_value:
                    offer[QUANTITY] = quantity_issue.max_value
        else:
            #print('買う'+str(self.q_need))
            opponent = ami.annotation["seller"]
            if len(self.q_opp_offer.keys()) >= 1:
                resource = sum(self.q_opp_offer.values()) #買う時の残りの数=相手が前回提示した数量の合計
                #prediction_resource = int(resource * 0.7) #相手の取引を考慮したうえでの残りの数
                fake_needs = resource / len(self.q_opp_offer.keys())
                offer[QUANTITY] = int(fake_needs) if fake_needs >= 1 else 1
        #print(offer)
        return tuple(offer)
    
    def best_q(self, opponent, my_need):
        offers_sorted = sorted(self.q_opp_offer.items(), key = lambda x : x[1], reverse=True)
        opponents = list(dict(offers_sorted).keys())
        q_opp_needs = list(dict(offers_sorted).values())
        if q_opp_needs[0] >= my_need:
            if opponent == opponents[0]:
                return my_need
            else:
                return 0
        while my_need < sum(q_opp_needs) and len(opponents) >= 2:
            opponents.remove(opponents[len(opponents) - 1])
            q_opp_needs.remove(q_opp_needs[len(opponents) - 1])
        if opponent in opponents:
            return int(my_need/len(opponents))
        else:
            return 0
    
    def respond(self, negotiator_id, state, source=""): #CCAgent
        #update q_opp_offer
        offer = state.current_offer
        # q_need = self._needed(negotiator_id)
        ami = self.get_nmi(negotiator_id)
        q = offer[QUANTITY]

        if self._is_selling(ami):
            opponent = ami.annotation["buyer"]
            self.q_opp_offer[opponent] = q
        else:
            opponent = ami.annotation["seller"]
            self.q_opp_offer[opponent] = q
        
        #response (compromise)
        if self.q_need <= 0:
            return ResponseType.END_NEGOTIATION

        else:
            if q <= self.q_need + 1:
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER
            
    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)
        #update q_need
        if self.is_seller:
            opponent = contract.annotation["buyer"]
        else:
            opponent = contract.annotation["seller"]
        if opponent in self.q_opp_offer:
            self.q_opp_offer.pop(opponent)
        #print(self.is_seller)
        self.q_need -= contract.agreement["quantity"]


class BetterAgent(SimpleAgent):
    """A greedy agent based on OneShotAgent with more sane strategy"""

    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state, source)
        if response != ResponseType.ACCEPT_OFFER:
            return response
        nmi = self.get_nmi(negotiator_id)
        return (
            response
            if self._is_good_price(nmi, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

    def _is_good_price(self, nmi, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # a good price is one better than the threshold
        if self._is_selling(nmi):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, nmi):
        """Finds a good-enough price conceding linearly over time"""
        state = nmi.state
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(nmi):
            return int(mn + th * (mx - mn))
        else:
            return int(mx - th * (mx - mn))

    def _price_range(self, nmi):
        """Finds the minimum and maximum prices"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        return mn, mx

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e

class AdaptiveAgent(BetterAgent):
    """Considers best price offers received when making its decisions"""

    def before_step(self):
        self._best_selling, self._best_buying = 0.0, float("inf")

    def respond(self, negotiator_id, state, source=""):
        """Save the best price received"""
        offer = state.current_offer
        response = super().respond(negotiator_id, state, source)
        nmi = self.get_nmi(negotiator_id)
        if self._is_selling(nmi):
            self._best_selling = max(offer[UNIT_PRICE], self._best_selling)
        else:
            self._best_buying = min(offer[UNIT_PRICE], self._best_buying)
        return response

    def _price_range(self, nmi):
        """Limits the price by the best price received"""
        mn, mx = super()._price_range(nmi)
        if self._is_selling(nmi):
            mn = max(mn, self._best_selling)
        else:
            mx = min(mx, self._best_buying)
        return mn, mx

class LearningAgent(AdaptiveAgent):
    def __init__(
        self,
        *args,
        acc_price_slack=float("inf"),
        step_price_slack=0.0,
        opp_price_slack=0.0,
        opp_acc_price_slack=0.2,
        range_slack=0.04,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._acc_price_slack = acc_price_slack
        self._step_price_slack = step_price_slack
        self._opp_price_slack = opp_price_slack
        self._opp_acc_price_slack = opp_acc_price_slack
        self._range_slack = range_slack

    def init(self):
        """Initialize the quantities and best prices received so far"""
        super().init()
        self._best_acc_selling, self._best_acc_buying = 0.0, float("inf")
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))
        self._best_opp_acc_selling = defaultdict(float)
        self._best_opp_acc_buying = defaultdict(lambda: float("inf"))

    def step(self):
        """Initialize the quantities and best prices received for next step"""
        super().step()
        self._best_opp_selling = defaultdict(float)
        self._best_opp_buying = defaultdict(lambda: float("inf"))

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        super().on_negotiation_success(contract, mechanism)

        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            self._best_acc_selling = max(up, self._best_acc_selling)
            self._best_opp_acc_selling[partner] = max(
                up, self._best_opp_acc_selling[partner]
            )
        else:
            partner = contract.annotation["seller"]
            self._best_acc_buying = min(up, self._best_acc_buying)
            self._best_opp_acc_buying[partner] = min(
                up, self._best_opp_acc_buying[partner]
            )

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        # find the quantity I still need and end negotiation if I need nothing more
        response = super().respond(negotiator_id, state)
        # update my current best price to use for limiting concession in other
        # negotiations
        ami = self.get_nmi(negotiator_id)
        up = offer[UNIT_PRICE]
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            self._best_opp_selling[partner] = max(up, self._best_selling)
        else:
            partner = ami.annotation["seller"]
            self._best_opp_buying[partner] = min(up, self._best_buying)
        return response

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            mn = min(
                mx * (1 - self._range_slack),
                max(
                    [mn]
                    + [
                        p * (1 - slack)
                        for p, slack in (
                            (self._best_selling, self._step_price_slack),
                            (self._best_acc_selling, self._acc_price_slack),
                            (self._best_opp_selling[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_selling[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        else:
            partner = ami.annotation["seller"]
            mx = max(
                mn * (1 + self._range_slack),
                min(
                    [mx]
                    + [
                        p * (1 + slack)
                        for p, slack in (
                            (self._best_buying, self._step_price_slack),
                            (self._best_acc_buying, self._acc_price_slack),
                            (self._best_opp_buying[partner], self._opp_price_slack),
                            (
                                self._best_opp_acc_buying[partner],
                                self._opp_acc_price_slack,
                            ),
                        )
                    ]
                ),
            )
        return mn, mx
    
#tournament
pd.options.display.float_format = '{:,.2f}'.format

def shorten_names(results):
    # just make agent types more readable
    results.score_stats.agent_type = results.score_stats.agent_type.str.split(".").str[-1]
    results.kstest.a = results.kstest.a.str.split(".").str[-1]
    results.kstest.b = results.kstest.b.str.split(".").str[-1]
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(".").str[-1]
    results.scores.agent_type = results.scores.agent_type.str.split(".").str[-1]
    results.winners = [_.split(".")[-1] for _ in results.winners]
    return results

###
###自分のエージェントを入れる(ベースエージェントからクラス継承)
###

tournament_types = [SimpleAgent, BetterAgent, AdaptiveAgent, LearningAgent] #自分のエージェント名を追加して実行
    # may take a long time
if __name__ == '__main__':
    results = anac2024_oneshot(
        competitors=tournament_types,
        n_configs=1, # number of different configurations to generate
        n_competitors_per_world=len(tournament_types),
        n_runs_per_world=10, # number of times to repeat every simulation (with agent assignment)
        n_steps=5, # number of days (simulation steps) per simulation
        print_exceptions=True,
        verbose = True,
    )

    results = shorten_names(results)

    print(len(results.scores.run_id.unique()))

    print(results.score_stats)

    results.scores["level"] = results.scores.agent_name.str.split("@", expand=True).loc[:, 1]
    results.scores = results.scores.sort_values("level")
    sns.lineplot(data=results.scores[["agent_type", "level", "score"]],
                x="level", y="score", hue="agent_type")
    plt.plot([0.0] * len(results.scores["level"].unique()), "b--")
    plt.show()
