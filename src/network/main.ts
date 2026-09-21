import { TransformerBlock } from "./block.ts";
import { argmax, randomMatrix, zeroVector } from "./common.ts";
import { EmbeddingLayer } from "./layers/embedding.ts";
import { PositionalEncoding } from "./layers/positional-encoding.ts";
import { Softmax } from "./layers/softmax.ts";
import { Linear } from "./linear.ts";
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
    private readonly outputLayer: Linear;
    private readonly softmax: Softmax;

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

            this.outputLayer = new Linear(
                randomMatrix(this.config.embeddingDim, this.config.vocabSize),
                zeroVector(this.config.vocabSize)
            );

        this.softmax = new Softmax();

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
        const eosId = this.tokenizer.getEndOfToken();

        const result: number[] = [];

        while (true) {
            let output = this.embedding.forward(tokens);
            output = this.positionalEncoding.forward(output);

            for (const block of this.blocks) {
                output = block.forward(output);
            }

            const logits = this.outputLayer.forward(output);
            const probabilities = this.softmax.forward(logits);
            const lastTokenPropabilities = probabilities[probabilities.length - 1];
            const nextToken = argmax(lastTokenPropabilities);
            
            if (nextToken == eosId) break;
            
            result.push(nextToken);
            tokens.push(nextToken);
        }

        return this.tokenizer.decode(result);
    }
}

const config: TransformerConfig = {
    vocabSize: 6,
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

const network = new Network(config, [",", "!", " ", "hello", "world", "<|eos|>"]);

const output = network.generateResponse(prompt("question?::::  ") ?? '');
console.log(output);
