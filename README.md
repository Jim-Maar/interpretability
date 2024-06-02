# interpretability

> #### Recap of the useful objects we've defined (4/4)
>
> Previously:
>
> * **`model` is an 8-layer autoregressive transformer.**
>     * It has been trained to predict legal Othello moves (all with the same probability).
>     * It gets fed a sequence of type `int` (i.e. integers from 0 to 60, where 0 represents "pass" (not present in our data) and the other numbers represent the 60 moves, excluding 4 middle squares).
> * **`board_seqs_int`, `board_seqs_string` are different representations of all 10000 of our games.**
>     * Both have shape `(num_games=10000, num_moves=60)`.
>     * The former has labels from 1 to 60, the latter from 0 to 63 excluding the middle squares.
> * **`focus_games_int`, `focus_games_string` - different representations of our "focus games".**
>     * Both have shape `(num_games=50, num_moves=60)`.
>     * The former has labels from 1 to 60, the latter from 0 to 63 excluding the middle squares.
> * **`focus_states` tells us what the board state is at any point.**
>     * It has shape `(num_games=50, num_moves=60, rows=8, cols=8)`, and the entries are 0, 1, -1 (for blank, black, white).
> * **`focus_valid_moves` tells us which moves are valid.**
>     * It has shape `(num_games=50, num_moves=60, rows*ncols=64)`, and the entries are 0, 1 (for illegal, legal).
> * **`focus_logits` and `focus_cache` - results of running our model on a bunch of games.**
>     * Shape of logits is `(num_games=50, num_moves=59, d_vocab=61)` (59 because we never predict the first move, 61 because we have 60 moves + 1 for pass).
>     * This gives us 3000 moves in total.
> * **`full_linear_probe` - tensor containing all the directions found by the trained probe**
>     * Has shape `(mode=3, d_model=512, row=8, col=8, options=3)`.
>         * `mode` refers to the mode in which the probe was trained: "odd moves / black to play", "even moves / white to play", and all moves.
>         * `options` refers to the three directions: "blank", "theirs", "mine".
>     * In other words, the inner product of `linear_probe[-1]` along its first dimension with the residual stream at some position gives us a tensor of shape `(rows=8, cols=8, options=3)` representing the probe's estimate for the model's logprobs for the options "blank", "black", "white" respectively.
> * **`linear_probe` - same as above, except:**
>     * It only has one mode ("odd move / black to play")
>         * So shape is `(d_model=512, row=8, col=8, options=3)`.
>     * Options are "blank", "theirs", "mine" respectively (because this turns out to be how the model actually thinks about the game).
>
> New:
>
> * **`blank_probe` and `my_probe` - tensors containing the important relative directions we care about in our analysis.**
>     * They were both created from linear combinations of the different **options** of `linear_probe`
>     * They both have shape `(d_model=512, row=8, col=8)`
>     * `blank_probe` represents the "blank direction" (which is `blank - 0.5 * theirs - 0.5 * mine`).
>     * `my_probe` represents the "mine - theirs direction" (which is `mine - theirs`).
>     * We can project residual stream vectors along these directions to get meaningful quantities (the model's "relative blank logprob", and "relative mine vs theirs logprob").
> * **`OthelloBoardState` is a class which is useful for creating board states over time.**
>     * You don't need to know how to use it; the code will always be given to you.
