// note: this is the simplest implementation i could possibly do 
//       so yes, it's bad how did u know

import { assert } from "@std/assert";

export class Tokenizer {
    constructor(private vocab: string[]) {}

    public tokenize(str: string) {
        const wordTokens = str.match(/\w+|[^\w\s]/g) ?? [];
        const output = [];
        for (const wordToken of wordTokens) {
            output.push(this.vocab.findIndex((t) => t == wordToken));
        }
        assert(!output.includes(-1));
        return output;
    } 
}
