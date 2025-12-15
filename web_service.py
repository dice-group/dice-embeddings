# Essential Design Patterns in Python
# A comprehensive guide with practical examples

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json


# =============================================================================
# 1. SINGLETON PATTERN
# =============================================================================
# Purpose: Ensure a class has only one instance and provide global access to it
# Use case: Database connections, logging, configuration settings
"""
Singleton Pattern

Use when: You need exactly one instance (database connections, loggers, caches)
Benefits: Memory efficiency, controlled access, global state
Drawbacks: Makes testing harder, can become a bottleneck, tight coupling
"""

class DatabaseConnection:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.connection_string = "database://localhost:5432"
            self.is_connected = False
            DatabaseConnection._initialized = True

    def connect(self):
        if not self.is_connected:
            print(f"Connecting to {self.connection_string}")
            self.is_connected = True
        return "Connected"

    def disconnect(self):
        if self.is_connected:
            print("Disconnecting from database")
            self.is_connected = False


# Pros: Controlled access to sole instance, reduced memory footprint
# Cons: Can make unit testing difficult, potential bottleneck, violates Single Responsibility Principle

# Example usage:
print("=== SINGLETON PATTERN ===")
db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(f"Same instance? {db1 is db2}")  # True
db1.connect()
print(f"DB2 connected? {db2.is_connected}")  # True (same instance)


# =============================================================================
# 2. FACTORY PATTERN
# =============================================================================
# Purpose: Create objects without specifying their exact classes
# Use case: Creating different types of objects based on input parameters
"""
Use when: Object creation logic is complex or you need to create different types based on parameters
Benefits: Loose coupling, easy to extend, centralized creation logic
Drawbacks: Can add unnecessary complexity for simple cases
"""
class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

    @abstractmethod
    def move(self) -> str:
        pass


class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

    def move(self) -> str:
        return "Running on four legs"


class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

    def move(self) -> str:
        return "Sneaking silently"


class Bird(Animal):
    def speak(self) -> str:
        return "Tweet!"

    def move(self) -> str:
        return "Flying in the sky"


class AnimalFactory:
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        animals = {
            'dog': Dog,
            'cat': Cat,
            'bird': Bird
        }

        animal_class = animals.get(animal_type.lower())
        if animal_class:
            return animal_class()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")


# Pros: Loose coupling, easy to extend with new types, centralized object creation
# Cons: Can become complex with many product types, may violate Open/Closed Principle

print("\n=== FACTORY PATTERN ===")
factory = AnimalFactory()
dog = factory.create_animal("dog")
cat = factory.create_animal("cat")
print(f"Dog says: {dog.speak()} and {dog.move()}")
print(f"Cat says: {cat.speak()} and {cat.move()}")


# =============================================================================
# 3. OBSERVER PATTERN
# =============================================================================
# Purpose: Define a one-to-many dependency between objects
# Use case: Event handling, model-view architectures, notifications
"""
Use when: Changes to one object require updating multiple dependent objects
Benefits: Loose coupling, dynamic relationships, supports broadcast communication
Drawbacks: Can cause unexpected cascading updates, memory leaks if not managed properly
"""
class Subject:
    def __init__(self):
        self._observers: List['Observer'] = []
        self._state = None

    def attach(self, observer: 'Observer'):
        self._observers.append(observer)

    def detach(self, observer: 'Observer'):
        self._observers.remove(observer)

    def notify(self):
        for observer in self._observers:
            observer.update(self)

    def set_state(self, state):
        self._state = state
        self.notify()

    def get_state(self):
        return self._state


class Observer(ABC):
    @abstractmethod
    def update(self, subject: Subject):
        pass


class EmailNotifier(Observer):
    def __init__(self, name: str):
        self.name = name

    def update(self, subject: Subject):
        print(f"Email to {self.name}: State changed to {subject.get_state()}")


class SMSNotifier(Observer):
    def __init__(self, phone: str):
        self.phone = phone

    def update(self, subject: Subject):
        print(f"SMS to {self.phone}: New state: {subject.get_state()}")


# Pros: Loose coupling between subject and observers, dynamic relationships
# Cons: Unexpected updates, memory leaks if observers aren't properly detached

print("\n=== OBSERVER PATTERN ===")
weather_station = Subject()
email_notifier = EmailNotifier("john@example.com")
sms_notifier = SMSNotifier("+1234567890")

weather_station.attach(email_notifier)
weather_station.attach(sms_notifier)
weather_station.set_state("Sunny, 25°C")
weather_station.set_state("Rainy, 18°C")


# =============================================================================
# 4. STRATEGY PATTERN
# =============================================================================
# Purpose: Define a family of algorithms and make them interchangeable
# Use case: Payment processing, sorting algorithms, data compression
"""
Use when: You have multiple ways to perform a task and want to switch between them
Benefits: Easy to add new algorithms, algorithms interchangeable at runtime
Drawbacks: Client must know about different strategies, more classes to maintain
"""
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount: float) -> str:
        pass


class CreditCardPayment(PaymentStrategy):
    def __init__(self, card_number: str):
        self.card_number = card_number

    def pay(self, amount: float) -> str:
        return f"Paid ${amount:.2f} using Credit Card ending in {self.card_number[-4:]}"


class PayPalPayment(PaymentStrategy):
    def __init__(self, email: str):
        self.email = email

    def pay(self, amount: float) -> str:
        return f"Paid ${amount:.2f} using PayPal account {self.email}"


class CryptoPayment(PaymentStrategy):
    def __init__(self, wallet_address: str):
        self.wallet_address = wallet_address

    def pay(self, amount: float) -> str:
        return f"Paid ${amount:.2f} using Crypto wallet {self.wallet_address[:10]}..."


class ShoppingCart:
    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self.payment_strategy: PaymentStrategy = None

    def add_item(self, name: str, price: float):
        self.items.append({"name": name, "price": price})

    def set_payment_strategy(self, strategy: PaymentStrategy):
        self.payment_strategy = strategy

    def checkout(self) -> str:
        if not self.payment_strategy:
            return "No payment method selected"

        total = sum(item["price"] for item in self.items)
        return self.payment_strategy.pay(total)


# Pros: Easy to add new algorithms, algorithms are interchangeable at runtime
# Cons: Clients must be aware of different strategies, increased number of classes

print("\n=== STRATEGY PATTERN ===")
cart = ShoppingCart()
cart.add_item("Laptop", 999.99)
cart.add_item("Mouse", 29.99)

# Try different payment methods
cart.set_payment_strategy(CreditCardPayment("1234567890123456"))
print(cart.checkout())

cart.set_payment_strategy(PayPalPayment("user@example.com"))
print(cart.checkout())


# =============================================================================
# 5. DECORATOR PATTERN
# =============================================================================
# Purpose: Add new functionality to objects without altering their structure
# Use case: Adding features to classes, middleware, logging
"""
Use when: You want to add responsibilities to objects without subclassing
Benefits: More flexible than inheritance, can combine multiple decorators
Drawbacks: Many small objects, can be hard to debug deep decorator chains
"""
class Coffee(ABC):
    @abstractmethod
    def cost(self) -> float:
        pass

    @abstractmethod
    def description(self) -> str:
        pass


class SimpleCoffee(Coffee):
    def cost(self) -> float:
        return 2.0

    def description(self) -> str:
        return "Simple coffee"


class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee

    def cost(self) -> float:
        return self._coffee.cost()

    def description(self) -> str:
        return self._coffee.description()


class MilkDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.5

    def description(self) -> str:
        return self._coffee.description() + ", milk"


class SugarDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.2

    def description(self) -> str:
        return self._coffee.description() + ", sugar"


class WhipDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.7

    def description(self) -> str:
        return self._coffee.description() + ", whipped cream"


# Pros: Add functionality without inheritance, compose behaviors at runtime
# Cons: Many small objects, can be complex to debug

print("\n=== DECORATOR PATTERN ===")
coffee = SimpleCoffee()
print(f"{coffee.description()}: ${coffee.cost():.2f}")

# Add decorators
coffee = MilkDecorator(coffee)
coffee = SugarDecorator(coffee)
coffee = WhipDecorator(coffee)
print(f"{coffee.description()}: ${coffee.cost():.2f}")


# =============================================================================
# 6. COMMAND PATTERN
# =============================================================================
# Purpose: Encapsulate a request as an object
# Use case: Undo/redo functionality, queuing operations, logging
"""
Use when: You need to queue operations, support undo/redo, or log requests
Benefits: Decouples sender from receiver, supports undo/redo, can queue/log commands
Drawbacks: Can be overkill for simple operations, increases number of classes
"""
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass


class Light:
    def __init__(self, location: str):
        self.location = location
        self.is_on = False

    def turn_on(self):
        self.is_on = True
        print(f"{self.location} light is ON")

    def turn_off(self):
        self.is_on = False
        print(f"{self.location} light is OFF")


class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self):
        self.light.turn_on()

    def undo(self):
        self.light.turn_off()


class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self):
        self.light.turn_off()

    def undo(self):
        self.light.turn_on()


class RemoteControl:
    def __init__(self):
        self.command_history: List[Command] = []

    def execute_command(self, command: Command):
        command.execute()
        self.command_history.append(command)

    def undo_last_command(self):
        if self.command_history:
            last_command = self.command_history.pop()
            last_command.undo()
        else:
            print("No commands to undo")


# Pros: Decouples sender from receiver, supports undo/redo, can log commands
# Cons: Increased number of classes, complexity for simple operations

print("\n=== COMMAND PATTERN ===")
living_room_light = Light("Living Room")
bedroom_light = Light("Bedroom")

remote = RemoteControl()

# Execute commands
remote.execute_command(LightOnCommand(living_room_light))
remote.execute_command(LightOnCommand(bedroom_light))
remote.execute_command(LightOffCommand(living_room_light))

# Undo commands
print("Undoing last commands:")
remote.undo_last_command()  # Turn living room light back on
remote.undo_last_command()  # Turn bedroom light off

print("\n=== SUMMARY ===")
print("Design patterns provide reusable solutions to common problems:")
print("• Singleton: One instance globally")
print("• Factory: Create objects without specifying exact classes")
print("• Observer: One-to-many dependencies")
print("• Strategy: Interchangeable algorithms")
print("• Decorator: Add functionality without inheritance")
print("• Command: Encapsulate requests as objects")

"""
General Guidelines
Pros of Design Patterns:

Provide tested, proven development solutions
Improve code readability and communication between developers
Make code more flexible and maintainable
Speed up development process

Cons of Design Patterns:

Can add unnecessary complexity if overused
May reduce performance in some cases
Can make code harder to understand for beginners
Sometimes used when simpler solutions would suffice

Best Practices:

Don't force patterns where they don't fit naturally
Start simple and add patterns when complexity justifies them
Consider the trade-offs for your specific use case
Remember that patterns are tools, not rules

Would you like me to dive deeper into any specific pattern or show you additional patterns like MVC, Adapter, or Facade?
"""