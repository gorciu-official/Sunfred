import { assert } from "@std/assert";

export function matmul(A: number[][], B: number[][]): number[][] {
    assert(A.length != 0 && B.length != 0);

    const aCols = A[0].length;
    const bCols = B[0].length;

    assert(aCols == B.length);

    const result: number[][] = Array.from(
        { length: A.length },
        () => Array(bCols).fill(0),
    );

    for (let i = 0; i < A.length; i++) {
        for (let k = 0; k < aCols; k++) {
            for (let j = 0; j < bCols; j++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
} 

export function randomMatrix(rows: number, cols: number): number[][] {
    const limit = Math.sqrt(
        6 / (rows + cols)
    );

    return Array.from(
        { length: rows },
        () =>
            Array.from(
                { length: cols },
                () =>
                    Math.random() * 2 * limit - limit
            )
    );
}

export function zeroVector(size: number): number[] {
    return new Array(size).fill(0);
}

export function addMatrices(A: number[][], B: number[][]) {
    return A.map((row, i) =>
        row.map((value, j) => value + B[i][j])
    );
}

export function relu(x: number) {
    return x > 0 ? x : 0;
}

export function splitHeads(
    x: number[][],
    numHeads: number
): number[][][] {
    const embeddingDim = x[0].length;
    const headDim = embeddingDim / numHeads;

    assert(Number.isInteger(headDim))

    const heads: number[][][] = [];

    for (let head = 0; head < numHeads; head++) {
        const start = head * headDim;
        const end = start + headDim;

        const headMatrix = x.map(row =>
            row.slice(start, end)
        );

        heads.push(headMatrix);
    }

    return heads;
}

export function concatHeads(heads: number[][][]): number[][] {
    if (heads.length === 0) {
        return [];
    }

    const seqLen = heads[0].length;

    return Array.from(
        { length: seqLen },
        (_, rowIndex) =>
            heads.flatMap(head => head[rowIndex])
    );
}
