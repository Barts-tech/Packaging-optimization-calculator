# Packaging Optimization Calculator

A Python-based tool designed to optimize the packing of pallets into shipping containers, considering their dimensions, weight, and stackability. The program uses the **bin packing algorithm** to efficiently distribute pallets into containers of varying sizes.

---

## Features

- **Pallet Input**: Enter pallet name, dimensions (length, width, height in centimeters), weight, quantity, and stackability.
- **Container Suggestions**: The program recommends suitable container types (20' or 40') based on the total weight and volume of the pallets.
- **Efficient Packing**: The bin packing algorithm ensures pallets are packed optimally to minimize unused space while respecting weight and volume limits.
- **Stackability**: Pallets that can be stacked are packed accordingly, maximizing available space.
- **Detailed Output**: The program provides a detailed breakdown of how pallets are packed into containers, including the number of containers used and pallet specifics.

---

## How It Works

1. **Input Pallet Data**:  
   Users input pallet details, such as:
   - Name
   - Dimensions (length, width, height in centimeters)
   - Weight (in kilograms)
   - Quantity
   - Stackability (whether the pallet can be stacked or not)
   
2. **Container Suggestion**:  
   Based on the total weight and volume of the pallets, the program suggests the most suitable container type (either 20' or 40').

3. **Packing Process**:  
   The pallets are packed into containers, ensuring the total weight and volume do not exceed container limits.

4. **Results**:  
   After packing, the program displays:
   - The number of containers used
   - A list of packed pallets, including quantity, weight, and volume

---

## Example

### User Input:

- **Pallet 1**:  
  - Name: `pallet_1`  
  - Dimensions: 100 cm x 120 cm x 80 cm  
  - Weight: 560 kg  
  - Quantity: 33 
  - Stackable: No

- **Pallet 2**:  
  - Name: `pallet_2`  
  - Dimensions: 150 cm x 150 cm x 120 cm  
  - Weight: 250 kg  
  - Quantity: 6 
  - Stackable: Yes



