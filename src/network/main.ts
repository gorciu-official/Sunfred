import { TransformerBlock } from "./block.ts";
import { EmbeddingLayer } from "./layers/embedding.ts";
import { PositionalEncoding } from "./layers/positional-encoding.ts";
import { Tokenizer } from "./tokenizer.ts";

type TransformerBlockConfig = {
    numHeads: number;
    hiddenDim: number;
};

type TransformerConfig = {
    vocabSize: number;
    embeddingDim: number;
    blocks: TransformerBlockConfig[];
};

class Network {
    private readonly tokenizer: Tokenizer;
    private readonly embedding: EmbeddingLayer;
    private readonly positionalEncoding: PositionalEncoding;
    private readonly blocks: TransformerBlock[];

    constructor(
        private readonly config: TransformerConfig,
        vocab: string[]
    ) {
        this.tokenizer = new Tokenizer(vocab);

        this.embedding = new EmbeddingLayer(
            config.embeddingDim,
            config.vocabSize
        );

        this.positionalEncoding =
            new PositionalEncoding(
                this.config.embeddingDim
            );

        this.blocks = this.config.blocks.map(
            blockConfig =>
                new TransformerBlock(
                    this.config.embeddingDim,
                    blockConfig.hiddenDim,
                    blockConfig.numHeads
                )
        );
    }

    generateResponse(question: string): string {
        const tokens = this.tokenizer.tokenize(question);

        let output = this.embedding.forward(tokens);
        output = this.positionalEncoding.forward(output);

        for (const block of this.blocks) {
            output = block.forward(output);
        }

        return JSON.stringify(output);
    }
}

const config: TransformerConfig = {
    vocabSize: 4,
    embeddingDim: 8,

    blocks: [
        {
            numHeads: 2,
            hiddenDim: 32,
        },
        {
            numHeads: 2,
            hiddenDim: 64,
        },
        {
            numHeads: 4,
            hiddenDim: 128,
        },
    ],
};

const network = new Network(config, [",", "!", "hello", "world"]);

const output = network.generateResponse("hello, world!");
console.log(output);
