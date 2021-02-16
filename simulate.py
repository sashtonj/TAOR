import simpy
import math
import random
from datetime import timedelta


"""
=== MOSEL VARIABLES ===
"""

global staff
global D

"""
=== PARAMETERS ===
"""

BUY_TICKET_TRANSACTION_TIME = 11  # Seconds
RELOAD_TRANSACTION_TIME = 17  # Seconds
TIME_BLOCK_LENGTH = 15 * 60  # 15 Minutes * 60 seconds
TIME_BLOCKS = 68  # Between 6am and 11pm
SIMULATION_RUNTIME = TIME_BLOCKS * TIME_BLOCK_LENGTH

"""
=== UTILITY FUNCTIONS ===
"""


def log(env, message, with_time=True):
    """
    Append the current simulation time to the print message. Add 6 hours because we start at 6am.
    """
    if with_time:
        print(f"{timedelta(seconds=env.now) + timedelta(hours=6)}: {message}")
    else:
        print(message)


def get_time_block(env):
    # Returns the current time block between [0, TIME_BLOCKS]
    return math.floor(env.now / TIME_BLOCK_LENGTH)


def parse_staff_schedule(staff):
    return [int(staff[x]) for x in list(staff.keys()) if x[0] == 1]


def parse_customer_demand(demands):
    return [int(demands[x]) for x in list(demands.keys()) if x[0] == 1]


"""
=== SIMULATION ===
"""


class Station(object):
    def __init__(self, env, schedule):
        self.env = env
        self.schedule = schedule
        self.on_shift = max(schedule)
        self.off_shift = []
        self.fare_collector = simpy.PriorityResource(self.env, capacity=self.on_shift)

    def buy_ticket(self, customer):
        yield self.env.timeout(BUY_TICKET_TRANSACTION_TIME)
        log(self.env, f"Customer {customer} bought a ticket!")

    def start_shift(self):
        """Ensures the correct number of staff are allocated at the start of each time block"""
        while True:
            log(self.env, f"Starting Time Block {get_time_block(self.env)}...", False)
            fare_collectors_required = self.on_shift - self.schedule[get_time_block(self.env)]
            if fare_collectors_required > 0:
                # Too many staff are currently working, remove the required amount
                self.env.process(self.remove_fare_collectors(fare_collectors_required))
            elif fare_collectors_required < 0:
                # Not enough staff working, sign-them the required amount back in
                self.add_fare_collectors(abs(fare_collectors_required))
            else:
                log(self.env, "No staff alterations")

            yield self.env.timeout(TIME_BLOCK_LENGTH)

    def remove_fare_collectors(self, num_to_remove):
        """
        To mimic the behaviour of removing staff, we request a fare-collector using a higher priority
        than customers (-1). Once we have the fare-collector, we don't release them back into the resource pool
        """
        for i in range(num_to_remove):
            request = self.fare_collector.request(priority=-1)
            yield request
            self.off_shift.append(request)
            self.on_shift -= 1
        log(self.env, f"Removed {num_to_remove} fare-collector(s)")

    def add_fare_collectors(self, num_to_add):
        """When we need more fare-collectors again, release them back into the pool to be used by customers"""
        log(self.env, f"Adding {num_to_add} fare-collector(s)")
        while num_to_add > 0:
            collector = self.off_shift.pop()
            self.fare_collector.release(collector)
            self.on_shift += 1
            num_to_add -= 1


def go_to_station(env, customer, station):
    log(env, f"Customer {customer} has arrived at the station")
    with station.fare_collector.request() as request:
        yield request
        yield env.process(station.buy_ticket(customer))


def simulate_customers(env, station, demands):
    # demands = [22, 97, 205, 439, 557, 797, 737, 814, 847, 809, 570, 608, 421, 461, 419, 349, 306, 262, 273, 282, 276,
    #            251, 216, 210, 193, 190, 204, 203, 228, 266, 296, 157, 151, 166, 182, 171, 179, 154, 170,
    #            159, 126, 111, 115, 107, 126, 111, 132, 122, 101, 115, 110, 99, 95, 71, 79, 84, 73, 78, 64, 47, 46, 31,
    #            44, 40, 31, 24, 14, 11]

    customer_idx = 1
    while True:
        time_between_customers = round(random.expovariate((demands[get_time_block(env)] * 60) / 54000))
        yield env.timeout(time_between_customers)
        env.process(go_to_station(env, customer_idx, station))
        customer_idx += 1


def start_simulation():
    # schedule = [1, 1, 2, 4, 5, 6, 6, 6, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2,
    #             2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    #             0, 0, 0, 0]
    #
    schedule = parse_staff_schedule(staff)
    demands = parse_customer_demand(D)
    env = simpy.Environment()
    station = Station(env, schedule)
    env.process(station.start_shift())
    env.process(simulate_customers(env, station, demands))
    env.run(until=SIMULATION_RUNTIME)


if __name__ == '__main__':
    start_simulation()
