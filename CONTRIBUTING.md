# Contributing to AutoEmulate

💫⚙️🤖 We're excited that you're here and want to contribute. 🤖⚙️💫

We want to ensure that every user and contributor feels welcome, included and supported to participate in the AutoEmulate community. Whether you're a seasoned developer, a machine learning researcher, a data scientist, or just someone eager to learn and contribute, **you are welcome here**. We value every contribution, be it big or small, and we appreciate the unique perspectives you bring to the project.

We hope that the information provided in this document will make it as easy as possible for you to get involved. If you find that you have questions that are not discussed below, please let us know through one of the many ways to [get in touch](#get-in-touch).

## Important Resources

If you'd like to find out more about AutoEmulate, make sure to check out:

1. **README**: For a high-level overview of the project, please refer to our [README](https://github.com/alan-turing-institute/autoemulate/blob/main/README.md).
2. **Documentation**: For more detailed information about the project, please refer to our [documentation](https://alan-turing-institute.github.io/autoemulate).

## How to Contribute

This section provides a high-level guide to contributing to AutoEmulate, designed for those with little or no experience with open source projects. For more detailed information, please also refer to the docs for:

* [contributing emulators](https://alan-turing-institute.github.io/autoemulate/community/contributing-emulators.html)
* [contributing to the docs](https://alan-turing-institute.github.io/autoemulate/community/contributing-docs.html)

We welcome contributions of all kinds, be it code, documentation, or community engagement. We encourage you to read through the following sections to learn more about how you can contribute to the package.

## Development guide                

### Running the test-suite on Apple silicon (M-series)

PyTorch’s Metal (MPS) backend still lacks some `float64` linear-algebra ops (e.g. `linalg_cholesky_ex`, `linalg_eigh`). On an M-series Mac these ops fail three Gaussian-process tests unless you let PyTorch fall back to CPU.

```# one-line fix: fall back to CPU for unsupported MPS ops
export PYTORCH_ENABLE_MPS_FALLBACK=1
pytest -q        # 831 passed, warnings only
```
If you prefer to skip the affected tests instead:

```
pytest -k "not mps"
```

## How to Submit Changes

We follow the same instructions for submitting changes to the project as those developed by [The Turing Way](https://github.com/the-turing-way/the-turing-way/blob/main/CONTRIBUTING.md#making-a-change-with-a-pull-request). In short, there are five steps to adding changes to this repository:

1. **Fork the Repository**: Start by [forking the AutoEmulate repository](https://github.com/alan-turing-institute/autoemulate/fork).
2. **Make Changes**: Ensure your code follows the existing code style ([PEP 8](https://peps.python.org/pep-0008/)) and passes all tests.
3. **Commit and Push**: Use clear commit messages.
4. **Open a Pull Request**: Ensure you describe the changes made and any additional details.

### 1. Fork the Repository

Once you have [created a fork of the repository](https://github.com/alan-turing-institute/autoemulate/fork), you now have your own unique local copy of AutoEmulate. Changes here won't affect anyone else's work, so it's a safe space to explore edits to the code!

Make sure to [keep your fork up to date](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) with the main repository, otherwise, you can end up with lots of dreaded [merge conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/about-merge-conflicts).

### 2. Make Changes

After writing new code or modifying existing code, please make sure to:

* write [numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
* write tests in the `tests/` directory using [pytest](https://docs.pytest.org/en/7.4.x/).
* format the code using [black](https://github.com/psf/black)

It would be great if you could also [update the documentation](https://alan-turing-institute.github.io/autoemulate/community/contributing-docs.html) to reflect the changes you've made. If you plan to add a new emulator have a look at the [contributing emulators docs](https://alan-turing-institute.github.io/autoemulate/community/contributing-emulators.html).

### 3. Commit and Push

While making your changes, commit often and write good, detailed commit messages. [This blog](https://chris.beams.io/posts/git-commit/) explains how to write a good Git commit message and why it matters.

#### Run pre-commit locally

We run [`pre-commit`](https://pre-commit.com/) in CI for every pull request, so please run it before you push to GitHub. This keeps formatting, linting, and metadata changes consistent and helps you catch failures earlier.

1. Install the hook runner once (pick the option that matches how you manage dependencies):
   * `pip install pre-commit`
   * or `uv tool install pre-commit`
2. Register the hooks in your clone so they run automatically on every `git commit`:
   ```
   pre-commit install
   ```
3. When you want to check the entire tree (recommended right before you open a PR), run:
   ```
   pre-commit run --all-files
   ```

If a hook makes changes, simply re-stage the affected files and run the command again until you get a `Passed` message. For hooks that fail with an error, follow the hint printed in the terminal (for example, formatting with Black or fixing lint) and then re-run `pre-commit run --all-files`.

### 4. Open a Pull Request

We encourage you to open a pull request as early in your contributing process as possible. This allows everyone to see what is currently being worked on. It also provides you, the contributor, feedback in real-time. GitHub has a [nice introduction](https://guides.github.com/introduction/flow) to the pull request workflow.

## First-timers' Corner

Just to-reiterate: We welcome all contributions, no matter how big or small! If anything in this guide is unclear, please reach out to ask or simply ask questions in a PR or issue.

## Reporting Bugs

Found a bug? Please open an issue here on GitHub to report it. We have a template for opening issues, so make sure you follow the correct format and ensure you include:

* A clear title.
* A detailed description of the bug.
* Steps to reproduce it.
* Expected versus actual behavior.

## Recognising Contributions

All contributors will be acknowledged in the [contributors](https://github.com/alan-turing-institute/autoemulate/tree/main#contributors) section of the README.

AutoEmulate follows the [all-contributors](https://github.com/kentcdodds/all-contributors#emoji-key) specifications. The all-contributors bot usage is described [here](https://allcontributors.org/docs/en/bot/usage).

To add yourself or someone else as a contributor, comment on the relevant Issue or Pull Request with the following:

> @all-contributors please add username for contribution1, contribution2

You can see the [Emoji Key (Contribution Types Reference)](https://allcontributors.org/docs/en/emoji-key) for a list of valid <contribution> types and examples of how this command can be run in [this issue](https://github.com/alan-turing-institute/autoemulate/issues/94). The bot will then create a Pull Request to add the contributor and reply with the pull request details.

**PLEASE NOTE: Only one contributor can be added with the bot at a time!** Add each contributor in turn, merge the pull request and delete the branch (`all-contributors/add-<username>`) before adding another one. Otherwise, you can end up with dreaded [merge conflicts](https://help.github.com/articles/about-merge-conflicts). Therefore, please check the open pull requests first to make sure there aren't any [open requests from the bot](https://github.com/alan-turing-institute/autoemulate/pulls/app%2Fallcontributors) before adding another.

What happens if you accidentally run the bot before the previous run was merged and you got those pesky merge conflicts? (Don't feel bad, we have all done it! 🙈) Simply close the pull request and delete the branch (`all-contributors/add-<username>`). If you are unable to do this for any reason, please <!-- let us know on Slack <link to Slack>--> reach out to us via email or open an issue, and one of our core team members will be very happy to help!

## Need Help?

If you're stuck or need assistance:

<!-- #TODO #148 - Check our [FAQ](<ADD LINK TO FAQ DOCUMENT>) section first. -->
* Reach out <!-- on Slack or --> via email for personalised assistance. (See [Get in touch](#get-in-touch) above for links.)
* Consider pairing up with a another contributor for guidance. <!-- You can always find us in the Slack channel and we're happy to chat! -->Contact us for guidance on this topic

**Once again, thank you for considering contributing to AutoEmulate! We hope you enjoy your contributing experience.**

## Inclusivity

We aim to make AutoEmulate a collaboratively developed project. We, therefore, require that all our members and their contributions **adhere to our [Code of Conduct](https://github.com/alan-turing-institute/autoemulate/blob/main/CODE_OF_CONDUCT.md)**. Please familiarise yourself with our Code of Conduct that lists the expected behaviours.

Every contributor is expected to adhere to our Code of Conduct. It outlines our expectations and ensures a safe, respectful environment for everyone.

----

These Contributing Guidelines have been adapted from the [Contributing Guidelines](https://github.com/the-turing-way/the-turing-way/blob/main/CONTRIBUTING.md#recognising-contributions) of [The Turing Way](https://github.com/the-turing-way/the-turing-way)! (License: CC-BY)

## Get in touch

Please visit our [GitHub repository](https://github.com/alan-turing-institute/autoemulate) and open an issue or discussion.

<!-- The easiest way to get involved with the active development of AutoEmulate is to join our regular community calls. The community calls are currently on a hiatus but if you are interested in participating in the forthcoming community calls, which will start in 2024, you should join our Slack workspace, where conversation about when to hold the community calls in the future will take place. -->

<!--
**Slack Workspace**: Join our [AutoEmulate Slack channel](<LINK TO SIGN-UP OR TO THE SLACK TEAM>) for discussions, queries, and community interactions. Send us an email at kwesterling@turing.ac.uk to request an invite.
-->
