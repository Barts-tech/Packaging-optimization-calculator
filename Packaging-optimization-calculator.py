# packaging_optimization_calculator.py

class Container:
    def __init__(self, max_weight, max_volume, container_type):
        self.max_weight = max_weight
        self.max_volume = max_volume
        self.container_type = container_type
        self.current_weight = 0
        self.current_volume = 0
        self.pallets = []

    def can_pack(self, pallet, quantity):
        total_weight = self.current_weight + pallet['weight'] * quantity
        total_volume = self.current_volume + pallet['volume'] * quantity
        return total_weight <= self.max_weight and total_volume <= self.max_volume

    def pack(self, pallet, quantity):
        if self.can_pack(pallet, quantity):
            self.pallets.append((pallet, quantity))
            self.current_weight += pallet['weight'] * quantity
            self.current_volume += pallet['volume'] * quantity
            return True
        return False

class BinPackingCalculator:
    def __init__(self, container_max_weight, container_max_volume, container_type):
        self.container_max_weight = container_max_weight
        self.container_max_volume = container_max_volume
        self.container_type = container_type
        self.containers = []

    def pack_pallets(self, pallets):
        pallets.sort(key=lambda x: (x['volume'], x['weight']), reverse=True)

        for pallet in pallets:
            quantity_remaining = pallet['quantity']
            while quantity_remaining > 0:
                packed = False
                # Try to pack into existing containers
                for container in self.containers:
                    if container.pack(pallet, 1):
                        quantity_remaining -= 1
                        packed = True
                        break

                # If packing into an existing container failed, create a new container
                if not packed:
                    new_container = Container(self.container_max_weight, self.container_max_volume, self.container_type)
                    if new_container.pack(pallet, 1):
                        self.containers.append(new_container)
                        quantity_remaining -= 1

    def display_results(self):
        if len(self.containers) == 0:
            print("No containers were packed.")
        else:
            print(f'Packed into {len(self.containers)} containers:')
            for i, container in enumerate(self.containers, start=1):
                print(f' Container {i} ({container.container_type}) - {len(container.pallets)} pallets')
                for pallet, quantity in container.pallets:
                    print(f'   Pallet: {pallet["name"]}, Quantity: {quantity}, Weight: {pallet["weight"]} kg, Volume: {pallet["volume"]} m³')

# Function to retrieve pallet data
def get_pallet_data():
    name = input("Enter pallet name: ")
    length_cm = float(input(f"Enter length of {name} (in centimeters): "))  # Length cm
    width_cm = float(input(f"Enter width of {name} (in centimeters): "))  # width cm
    height_cm = float(input(f"Enter height of {name} (in centimeters): "))  # height cm
    weight = float(input(f"Enter weight of {name} (in kilograms): "))  # weight kg
    quantity = int(input(f"Enter quantity of {name} pallets: "))  # quantity 
    stackable = input(f"Can pallet {name} be stacked? (yes/no): ").strip().lower() == 'yes'

    # Converting dimensions from centimeters to meters and calculating the volume of a pallet
    length_m = length_cm / 100
    width_m = width_cm / 100
    height_m = height_cm / 100
    volume = length_m * width_m * height_m  # Calculation of volume in cubic meters
    return {'name': name, 'length': length_m, 'width': width_m, 'height': height_m, 'weight': weight, 'volume': volume, 'quantity': quantity, 'stackable': stackable}

def get_container_data():
    max_weight = float(input("Enter maximum weight capacity of the container (in kilograms): "))
    max_volume = float(input("Enter maximum volume capacity of the container (in cubic meters): "))  # Capacity in m³
    return max_weight, max_volume

# Function to suggest a container based on weight and volume
def suggest_container_type(total_weight, total_volume):
    if total_weight <= 22000 and total_volume <= 33:
        return {'type': '20\'', 'max_weight': 22000, 'max_volume': 33}
    elif total_weight <= 40000 and total_volume <= 67:
        return {'type': '40\'', 'max_weight': 40000, 'max_volume': 67}
    else:
        return None

# Main function
def main():
    print("Welcome to the Bin Packing Calculator!")

    # User enters data
    pallets = []
    
    while True:
        pallet = get_pallet_data()
        pallets.append(pallet)
        
        # Ask the user if he wants to add another palette
        more_pallets = input("Do you want to add another pallet? (yes/no): ").strip().lower()
        if more_pallets in ['no', 'n']:
            break

    # Calculation of total weight and volume of pallets
    total_weight = sum(pallet['weight'] * pallet['quantity'] for pallet in pallets)
    total_volume = sum(pallet['volume'] * pallet['quantity'] for pallet in pallets)

    # Suggesting the type of container
    suggested_container = suggest_container_type(total_weight, total_volume)

    if suggested_container:
        print(f"\nSuggested container: {suggested_container['type']} (Max weight: {suggested_container['max_weight']} kg, Max volume: {suggested_container['max_volume']} m³)")
        container_max_weight, container_max_volume = suggested_container['max_weight'], suggested_container['max_volume']
    else:
        print("\nNo suitable container found based on the total weight and volume.")
        # deploy pallets into smaller containers if there is no suitable container available
        container_max_weight, container_max_volume = 22000, 33  
        print("We will divide the pallets into smaller containers. Here is the plan:")

    # Initialization of the packing calculator
    calculator = BinPackingCalculator(container_max_weight, container_max_volume, suggested_container['type'] if suggested_container else '20\'')

    # Pallet packing
    calculator.pack_pallets(pallets)

    
    calculator.display_results()


if __name__ == "__main__":
    main()
