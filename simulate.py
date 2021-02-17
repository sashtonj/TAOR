import simpy
import math
import random
import numpy as np
from datetime import timedelta
from enum import IntEnum


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


class Collector(IntEnum):
    WITH_TERMINAL = 0
    WITHOUT_TERMINAL = 1


class PurchaseType(IntEnum):
    BUY = 0
    RELOAD = 1


class FareCollector(simpy.PriorityResource):
    def __init__(self, env, capacity, has_terminal=False):
        self.env = env
        self.on_shift = False
        self.collector_type = Collector.WITH_TERMINAL if has_terminal else Collector.WITHOUT_TERMINAL
        super().__init__(env, capacity)

    def sell_ticket(self, customer):
        yield self.env.timeout(BUY_TICKET_TRANSACTION_TIME)
        log(self.env, f"Customer {customer} bought a ticket!")

    def reload_card(self, customer):
        yield self.env.timeout(RELOAD_CARD_TRANSACTION_TIME)
        log(self.env, f"Customer {customer} reloaded their card!")

    def get_queue_length(self):
        return len(self.queue) + len(self.users)


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
    """Translates the mosel staff schedule into a 2D array in the format [collector_type, time_block]"""
    schedule = np.zeros((2, TIME_BLOCKS), dtype=int)
    for collector, time_block in list(staff_schedule.keys()):
        schedule[collector - 1, time_block - 1] = staff_schedule[collector, time_block]
    return schedule


def parse_customer_demand(demand):
    """Translates the mosel demands into a 2D array in the format [purchase_type, time_block]"""
    demands = np.zeros((2, TIME_BLOCKS), dtype=int)
    for purchase_type, time_block in list(demand.keys()):
        demands[purchase_type - 1, time_block - 1] = demand[purchase_type, time_block]
    return demands


"""
=== SIMULATION ===
"""


class Station(object):
    def __init__(self, env, schedule):
        self.env = env
        self.schedule = schedule
        self.booth = []
        for i in range(max(self.schedule[Collector.WITH_TERMINAL])):
            self.booth.append(FareCollector(self.env, capacity=1, has_terminal=True))

        for i in range(max(self.schedule[Collector.WITHOUT_TERMINAL])):
            self.booth.append(FareCollector(self.env, capacity=1))

    def count_collectors_on_shift(self, collector_type):
        return len([x for x in self.booth if x.collector_type is collector_type and x.on_shift])

    def allocate_staff(self, collector_type):
        on_shift = self.count_collectors_on_shift(collector_type)
        required = on_shift - self.schedule[collector_type][get_time_block(self.env)]
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

            self.allocate_staff(Collector.WITH_TERMINAL)
            self.allocate_staff(Collector.WITHOUT_TERMINAL)

            # Wait the length of one time block
            yield self.env.timeout(TIME_BLOCK_LENGTH)

    def remove_fare_collectors(self, num_to_remove, collector_type):
        """
        To mimic the behaviour of removing staff, we request a fare-collector using a higher priority
        than customers (-1). Once we have the fare-collector, we don't release them back into the resource pool
        """
        on_shift = list(filter(lambda x: x.collector_type is collector_type and x.on_shift, self.booth))
        for i in range(num_to_remove):
            with on_shift[i].request(priority=-1) as request:
                yield request
                on_shift[i].on_shift = False
                # Clear get queue
        log(self.env, f"Removed {num_to_remove} fare-collector(s) of type {collector_type}")

    def add_fare_collectors(self, num_to_add, collector_type):
        """When we need more fare-collectors again, release them back into the pool to be used by customers"""
        log(self.env, f"Adding {num_to_add} fare-collector(s) of type {collector_type}")
        on_shift = list(filter(lambda x: x.collector_type is collector_type and not x.on_shift, self.booth))
        for x in range(num_to_add):
            on_shift[x].on_shift = True


def select_fare_collector(station, purchase_type):
    """Customers always select the fare-collector with the shortest queue. Return the selected fare-collector"""
    # Get all fare collectors currently on shift
    on_shift = list(filter(lambda x: x.on_shift, station.booth))
    if purchase_type is PurchaseType.RELOAD:
        # Customer is reloading their prepaid card, therefore the fare-collector must have a terminal
        # Filter out staff that don't have a terminal
        on_shift = list(filter(lambda x: x.collector_type is Collector.WITH_TERMINAL, on_shift))
    queue_lengths = list(map(lambda x: x.get_queue_length(), on_shift))
    return on_shift[queue_lengths.index(min(queue_lengths))]


def go_to_station(env, customer, station, purchase_type):
    log(env, f"Customer {customer} has arrived at the station")
    fare_collector = select_fare_collector(station, purchase_type)
    with fare_collector.request() as request:
        yield request
        if purchase_type is PurchaseType.RELOAD:
            yield env.process(fare_collector.reload_card(customer))
        elif purchase_type is PurchaseType.BUY:
            yield env.process(fare_collector.sell_ticket(customer))


def simulate_customers(env, station, demands, purchase_type):
    customer_idx = 1 if purchase_type is PurchaseType.BUY else 2
    while True:
        arrivals_per_second = calculate_arrival_rate(demands[get_time_block(env)])
        time_between_customers = round(random.expovariate(arrivals_per_second))
        yield env.timeout(time_between_customers)
        env.process(go_to_station(env, customer_idx, station, purchase_type))
        customer_idx += 2


def start_simulation():
    # Get data from optimization model
    schedule = parse_staff_schedule(staff)
    demands = parse_customer_demand(D)

    # Initialise simulation environment
    env = simpy.Environment()
    station = Station(env, schedule)
    # Allocate staff
    env.process(station.start_shift())
    # Start simulating customers with prepaid cards
    env.process(simulate_customers(env, station, demands[PurchaseType.RELOAD], PurchaseType.RELOAD))
    # Start simulating customers buying tickets
    env.process(simulate_customers(env, station, demands[PurchaseType.BUY], PurchaseType.BUY))

    env.run(until=SIMULATION_RUNTIME)


if __name__ == '__main__':
    start_simulation()
