export class PositionalEncoding {
    constructor(private readonly embeddingDim: number) {}

    forward(x: number[][]): number[][] {
        const result = x.map(row => [...row]);

        for (let pos = 0; pos < x.length; pos++) {
            for (let dim = 0; dim < this.embeddingDim; dim++) {
                const angle = 
                    pos / Math.pow(
                        10000, (2 * Math.floor(dim / 2)) / this.embeddingDim
                    );

                const positionalValue = (dim % 2 === 0)
                    ? Math.sin(angle)
                    : Math.cos(angle)

                result[pos][dim] += positionalValue;
            }
        }

        return result;
    }
}
