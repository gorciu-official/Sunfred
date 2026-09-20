export class LayerNorm {
    // no, it's not https://github.com/archivepedia/epsilon (peak project btw)
    private readonly epsilon = 1e-5;

    public forward(x: number[][]): number[][] {
        return x.map((row) => {
            const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
            const variance = row.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / row.length;
            return row.map(val => {
                return (val - mean) / Math.sqrt(variance + this.epsilon)
            });
        })
    }
}
