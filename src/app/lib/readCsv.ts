function parseCSVLine(line: string) {
    const result = [];
    let current = '';
    let inQuotes = false;
    let i = 0;

    while (i < line.length) {
        const char = line[i];

        if (inQuotes) {
            if (char === '"') {
                if (line[i + 1] === '"') {
                    current += '"'; // Escaped quote
                    i++;
                } else {
                    inQuotes = false; // End of quoted field
                }
            } else {
                current += char;
            }
        } else {
            if (char === '"') {
                inQuotes = true;
            } else if (char === ',') {
                result.push(convertValue(current));
                current = '';
            } else {
                current += char;
            }
        }
        i++;
    }

    // Push the last value
    if (current !== '' || line.endsWith(',')) {
        result.push(convertValue(current));
    }

    return result;
}

// Helper to convert numeric strings to numbers
function convertValue(value: string) {
    const trimmed = value.trim();
    const num = Number(trimmed);
    return trimmed !== '' && !isNaN(num) ? num : trimmed;
}

/**
 * Reads a CSV file from a File object and returns it as an array of arrays.
 * @param file The File object (from input[type="file"]).
 * @returns Promise resolving to array of rows, where each row is an array of strings or numbers.
 */
export function readCsv(file: File): Promise<(string | number)[][]> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const content = reader.result as string;
            const lines = content.trim().split('\n');
            const result = lines.map((line) => parseCSVLine(line));
            resolve(result);
        };
        reader.onerror = reject;
        reader.readAsText(file);
    });
}
