/**
 * Encodes an object (Record<string, number | number[] | number[][]>) into a Float32Array.
 * The format is now:
 * [numberOfEntries,
 * ...[entry1KeyLength, ...entry1KeyBytes, entry1ValueCols, entry1ValueRows, ...entry1ValueData],
 * ...[entry2KeyLength, ...entry2KeyBytes, entry2ValueCols, entry2ValueRows, ...entry2ValueData],
 * ...]
 *
 * Value dimension values (for individual entry values):
 * - Single number: valueCols = -1, valueRows = -1
 * - 1D array: valueCols = length, valueRows = -1
 * - 2D array: valueCols = num_cols, valueRows = num_rows
 *
 * @param data The object data (Record<string, number | number[] | number[][]>).
 * @returns A Float32Array containing the encoded data.
 */
export const encode = (
    data: Record<string, number | number[] | number[][] | undefined>,
): Float32Array => {
    const objectEntries = Object.entries(data).filter(
        ([, value]) => value !== undefined && value !== null,
    );
    const numberOfEntries = objectEntries.length;
    let allFlattenedEntryData: number[] = [];

    // Iterate through each key-value pair in the object
    for (const [key, value] of objectEntries) {
        // Encode the key (sub-field name)
        const keyBytes = key.split('').map((char) => char.charCodeAt(0));
        const keyLength = keyBytes.length;

        let valueCols = -1;
        let valueRows = -1;
        let flattenedValue: number[] = [];

        // Determine the type and flatten the value data
        if (typeof value === 'number') {
            // Single number
            flattenedValue = [value];
            valueCols = -1; // Sentinel for single number
            valueRows = -1; // Sentinel for single number
        } else if (Array.isArray(value)) {
            if (value.length === 0) {
                // Empty array, treat as 1D with 0 length
                valueCols = 0;
                valueRows = -1;
                flattenedValue = [];
            } else if (typeof value[0] === 'number') {
                // 1D array
                flattenedValue = value as number[];
                valueCols = flattenedValue.length;
                valueRows = -1; // Sentinel for 1D array
            } else if (Array.isArray(value[0])) {
                // 2D array
                const value2D = value as number[][];
                valueRows = value2D.length;
                if (valueRows > 0) {
                    valueCols = value2D[0].length;
                    // Flatten the 2D array
                    flattenedValue = value2D.flat();
                } else {
                    valueCols = 0; // Empty 2D array
                    valueRows = 0; // Empty 2D array
                }
            } else {
                throw new Error(
                    `Unsupported array element type ${typeof value[0]} for key '${key}'. Array must contain numbers or number arrays.`,
                );
            }
        } else {
            throw new Error(
                `Unsupported value type ${typeof value} for key '${key}'. Value must be number, number[], or number[][].`,
            );
        }

        // Add the encoded key and value data to the main flattened data array
        allFlattenedEntryData.push(keyLength);
        allFlattenedEntryData.push(...keyBytes);
        allFlattenedEntryData.push(valueCols, valueRows);

        // Use concat for large arrays to avoid stack overflow
        if (flattenedValue.length > 10000) {
            allFlattenedEntryData = allFlattenedEntryData.concat(flattenedValue);
        } else {
            allFlattenedEntryData.push(...flattenedValue);
        }
    }

    // Create Float32Array with numberOfEntries as the first element
    return new Float32Array([numberOfEntries, ...allFlattenedEntryData]);
};

/**
 * Decodes a Float32Array back into an object.
 *
 * @param encodedData The Float32Array to decode.
 * @returns An object containing the decoded data.
 */
export const decode = <T = Record<string, number | number[] | number[][]>>(
    encodedData: Float32Array,
): T => {
    let offset = 0;
    const decodedObject: Record<string, number | number[] | number[][]> = {};

    if (encodedData.length === 0) {
        throw new Error('Encoded data is empty.');
    }

    // Read numberOfEntries from the first element
    const numberOfEntries = encodedData[offset++];
    if (isNaN(numberOfEntries) || numberOfEntries < 0) {
        throw new Error('Invalid number of entries at the beginning of the encoded data.');
    }

    // Loop through the encoded data for each entry
    for (let i = 0; i < numberOfEntries; i++) {
        // Read key length
        if (offset >= encodedData.length) {
            throw new Error(`Encoded data truncated: missing key length for entry ${i}.`);
        }
        const keyLength = encodedData[offset++];
        if (isNaN(keyLength) || keyLength < 0 || offset + keyLength > encodedData.length) {
            throw new Error(
                `Invalid key length for entry ${i} at offset ${offset - 1} or truncated data.`,
            );
        }

        // Read key characters
        const keyBytes = Array.from(encodedData.slice(offset, offset + keyLength));
        const key = String.fromCharCode(...keyBytes);
        offset += keyLength;

        // Read value dimensions (cols and rows for the current value)
        if (offset + 2 > encodedData.length) {
            throw new Error(
                `Encoded data truncated: missing value dimensions for key '${key}' at offset ${offset}.`,
            );
        }
        const valueCols = encodedData[offset++];
        const valueRows = encodedData[offset++];

        let value: number | number[] | number[][];

        // Reconstruct the value based on its dimensions
        if (valueCols === -1 && valueRows === -1) {
            // Single number
            if (offset >= encodedData.length) {
                throw new Error(
                    `Encoded data truncated: missing single number value for key '${key}' at offset ${offset}.`,
                );
            }
            value = encodedData[offset++];
        } else if (valueCols !== -1 && valueRows === -1) {
            // 1D array
            const expectedLength = valueCols;
            if (offset + expectedLength > encodedData.length) {
                throw new Error(
                    `Encoded data truncated: missing 1D array data for key '${key}' at offset ${offset}. Expected ${expectedLength} elements.`,
                );
            }
            value = Array.from(encodedData.slice(offset, offset + expectedLength));
            offset += expectedLength;
        } else if (valueCols !== -1 && valueRows !== -1) {
            // 2D array
            const expectedLength = valueCols * valueRows;
            if (offset + expectedLength > encodedData.length) {
                throw new Error(
                    `Encoded data truncated: missing 2D array data for key '${key}' at offset ${offset}. Expected ${expectedLength} elements.`,
                );
            }
            const rawData = encodedData.slice(offset, offset + expectedLength);
            offset += expectedLength;

            value = Array.from({ length: valueRows }, (_, i) =>
                Array.from(rawData.slice(i * valueCols, (i + 1) * valueCols)),
            );
        } else {
            throw new Error(
                `Invalid dimension information for key '${key}' at offset ${offset}. Cols: ${valueCols}, Rows: ${valueRows}.`,
            );
        }

        decodedObject[key] = value;
    }

    // Check if there's any unexpected data left
    if (offset < encodedData.length) {
        console.warn(
            `Warning: Extra data found in encoded array after decoding all entries. Remaining elements: ${encodedData.length - offset}`,
        );
    }

    // Since there's no root object field name, provide a default
    return decodedObject as T;
};
