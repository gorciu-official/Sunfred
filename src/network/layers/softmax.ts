export class Softmax {
    forward(x: number[][]): number[][] {
        return x.map(row => {
            const max = Math.max(...row);

            const exponentials = row.map(
                value => Math.exp(value - max)
            );

            const sum = exponentials.reduce(
                (acc, value) => acc + value,
                0
            );

            return exponentials.map(
                value => value / sum
            );
        });
    }
}
