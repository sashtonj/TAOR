import simpy
import math
import random
import numpy as np
from datetime import timedelta
from enum import Enum

"""
=== MOSEL VARIABLES ===
"""

global staff
global D

"""
=== PARAMETERS ===
"""

BUY_TICKET_TRANSACTION_TIME = 11  # Seconds
RELOAD_CARD_TRANSACTION_TIME = 17  # Seconds
TIME_BLOCK_LENGTH = 15 * 60  # 15 Minutes * 60 seconds
TIME_BLOCKS = 68  # Between 6am and 11pm
SIMULATION_RUNTIME = TIME_BLOCKS * TIME_BLOCK_LENGTH
VERBOSE = True  # Log output to the console?

"""
=== UTILITY FUNCTIONS/CLASSES ===
"""


class CollectorType(Enum):
    WITH_TERMINAL = 1
    WITHOUT_TERMINAL = 2


class PurchaseType(Enum):
    BUY = 1
    RELOAD = 2


def log(env, message, with_time=True):
    """
    Append the current simulation time to the print message. Add 6 hours because we start at 6am.
    """
    if VERBOSE:
        if with_time:
            print(f"{timedelta(seconds=env.now) + timedelta(hours=6)}: {message}")
        else:
            print(message)


def get_time_block(env):
    """Returns the current time block between [0, TIME_BLOCKS]"""
    return math.floor(env.now / TIME_BLOCK_LENGTH)


def calculate_arrival_rate(time_block_demand):
    """Calculates the customer arrival rate per second for the given time block"""
    return (time_block_demand * 60) / 54000


def parse_staff_schedule(staff_schedule):
    """Translates the mosel staff schedule into a 2D array [collector_type, time_block]"""
    schedule = np.zeros((2, TIME_BLOCKS), dtype=int)
    for collector, time_block in list(staff_schedule.keys()):
        schedule[collector - 1, time_block - 1] = staff_schedule[collector, time_block]
    return schedule


def parse_customer_demand(demand):
    """Translates the mosel demands into a 2D array [ticket_type, time_block]"""
    demands = np.zeros((2, TIME_BLOCKS), dtype=int)
    for ticket_type, time_block in list(demand.keys()):
        demands[ticket_type - 1, time_block - 1] = demand[ticket_type, time_block]
    return demands


"""
=== SIMULATION ===
"""


class Station(object):
    def __init__(self, env, schedule):
        self.env = env
        self.schedule = schedule
        self.on_shift = [max(schedule[CollectorType.WITH_TERMINAL]), max(schedule[CollectorType.WITHOUT_TERMINAL])]
        self.off_shift = []
        self.fare_collectors = [
            simpy.PriorityResource(self.env, capacity=self.on_shift[CollectorType.WITH_TERMINAL]),
            simpy.PriorityResource(self.env, capacity=self.on_shift[CollectorType.WITHOUT_TERMINAL])
        ]

    def buy_ticket(self, customer):
        yield self.env.timeout(BUY_TICKET_TRANSACTION_TIME)
        log(self.env, f"Customer {customer} bought a ticket!")

    def reload_card(self, customer):
        yield self.env.timeout(RELOAD_CARD_TRANSACTION_TIME)
        log(self.env, f"Customer {customer} reloaded their card!")

    def allocate_staff(self, collector_type):
        required = self.on_shift[collector_type] - self.schedule[collector_type]
        if required > 0:
            # Too many staff are currently working, remove the required amount
            self.env.process(self.remove_fare_collectors(required, collector_type))
        elif required < 0:
            #  Not enough staff working, add the required amount back in
            self.add_fare_collectors(abs(required), collector_type)

    def start_shift(self):
        """Ensures the correct number of staff are allocated at the start of each time block"""
        while True:
            log(self.env, f"Starting Time Block {get_time_block(self.env)}...", False)
            self.allocate_staff(CollectorType.WITH_TERMINAL)
            self.allocate_staff(CollectorType.WITHOUT_TERMINAL)

            # Wait the length of one time block
            yield self.env.timeout(TIME_BLOCK_LENGTH)

    def remove_fare_collectors(self, num_to_remove, collector_type):
        """
        To mimic the behaviour of removing staff, we request a fare-collector using a higher priority
        than customers (-1). Once we have the fare-collector, we don't release them back into the resource pool
        """
        for i in range(num_to_remove):
            request = self.fare_collectors[collector_type].request(priority=-1)
            yield request
            self.off_shift[collector_type].append(request)
            self.on_shift[collector_type] -= 1
        log(self.env, f"Removed {num_to_remove} fare-collector(s)")

    def add_fare_collectors(self, num_to_add, collector_type):
        """When we need more fare-collectors again, release them back into the pool to be used by customers"""
        log(self.env, f"Adding {num_to_add} fare-collector(s)")
        while num_to_add > 0:
            collector = self.off_shift[collector_type].pop()
            self.fare_collectors[collector_type].release(collector)
            self.on_shift[collector_type] += 1
            num_to_add -= 1


def go_to_station(env, customer, station, ticket_type):
    log(env, f"Customer {customer} has arrived at the station")
    if ticket_type == PurchaseType.BUY:

        request = station.fare_collectors[CollectorType.WITH_TERMINAL].request()\
                  | station.fare_collectors[CollectorType.WITHOUT_TERMINAL].request()
        yield request
        yield env.process(station.buy_ticket(customer))
    elif ticket_type == PurchaseType.RELOAD:
        request = station.fare_collectors[CollectorType.WITH_TERMINAL].request()
        yield request
        yield env.process(station.reload_card(customer))
        station.fare_collectors[CollectorType]


def simulate_customers(env, station, demands):
    customer_idx = 1
    while True:
        arrivals_per_second = calculate_arrival_rate(demands[get_time_block(env)])
        time_between_customers = round(random.expovariate(arrivals_per_second))
        yield env.timeout(time_between_customers)
        env.process(go_to_station(env, customer_idx, station))
        customer_idx += 1


def start_simulation():
    schedule = parse_staff_schedule(staff)
    demands = parse_customer_demand(D)
    env = simpy.Environment()
    station = Station(env, schedule)
    env.process(station.start_shift())
    env.process(simulate_customers(env, station, demands[CollectorType.WITH_TERMINAL]))
    env.process(simulate_customers(env, station, demands[CollectorType.WITHOUT_TERMINAL]))
    env.run(until=SIMULATION_RUNTIME)


if __name__ == '__main__':
    start_simulation()
