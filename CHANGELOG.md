# Changelog

All notable changes to this project will be documented in this file. See
[Conventional Commits](https://conventionalcommits.org) for commit guidelines.

## [1.1.0](https://github.com/emma-simbot/policy/compare/v1.0.0...v1.1.0) (2022-10-28)


### Features

* Replace QA format with speaker tokens ([#616](https://github.com/emma-simbot/policy/issues/616)) ([895cb8d](https://github.com/emma-simbot/policy/commit/895cb8d65c03459b0581b276f0a7d80aa4756747))

## 1.0.0 (2022-10-27)


### Features

* Ban generating past frames ([#604](https://github.com/emma-simbot/policy/issues/604)) ([a356c6b](https://github.com/emma-simbot/policy/commit/a356c6bbb51ba4f472fc077dd135f9cb9cca419c))
* be able to change the number of target tokens per decoding step from the args ([#395](https://github.com/emma-simbot/policy/issues/395)) ([0782e71](https://github.com/emma-simbot/policy/commit/0782e719cf02276c0a6b359d503f85ebbd2e38f0))
* fix the image comparison to stop being false positive ([#378](https://github.com/emma-simbot/policy/issues/378)) ([558e1cb](https://github.com/emma-simbot/policy/commit/558e1cb903cd29f68f5400d9d2a6037dddfa6574))
* Subsample look and goto table/desk ([#607](https://github.com/emma-simbot/policy/issues/607)) ([09344bc](https://github.com/emma-simbot/policy/commit/09344bc444ab6bff18c126984ab9ee7910f7d957))


### Bug Fixes

* Add endpoint in run_model ([#615](https://github.com/emma-simbot/policy/issues/615)) ([53fa772](https://github.com/emma-simbot/policy/commit/53fa772e5135699f303ce005be60653fbb74efeb))
* bounding boxes ([#382](https://github.com/emma-simbot/policy/issues/382)) ([a834f5d](https://github.com/emma-simbot/policy/commit/a834f5dac35a70588ac3376283bd8938591e83df))
* dataloader types in emma pretrain datamodule ([#137](https://github.com/emma-simbot/policy/issues/137)) ([b317198](https://github.com/emma-simbot/policy/commit/b317198f6c0f831cec4098c4cfc266b26399347f))
* empty trajectory or action delimiter only ([#398](https://github.com/emma-simbot/policy/issues/398)) ([028cb94](https://github.com/emma-simbot/policy/commit/028cb9450fef071ea648bfcd3616adde1bfd7577))
* input encoding needs to be the same as the training ([#380](https://github.com/emma-simbot/policy/issues/380)) ([094b984](https://github.com/emma-simbot/policy/commit/094b98474b4c2c06162fa8509aeac1f677dfbf9f))
* original history length set to minumum ([#386](https://github.com/emma-simbot/policy/issues/386)) ([659721d](https://github.com/emma-simbot/policy/commit/659721d46948112936321d4dcaa66f6f50734c17))
* pick a random object index from the ones in the current frame ([#393](https://github.com/emma-simbot/policy/issues/393)) ([565beb5](https://github.com/emma-simbot/policy/commit/565beb526a1d1d02db35c4bee966f9ff3531c0fe))
* set the original history length to be max at the max frames ([#383](https://github.com/emma-simbot/policy/issues/383)) ([39df383](https://github.com/emma-simbot/policy/commit/39df3838f3dc029116da17de4ed3f9f8f8968cae))
* update the previous frame AFTER the comparison is made ([#381](https://github.com/emma-simbot/policy/issues/381)) ([6e8f979](https://github.com/emma-simbot/policy/commit/6e8f979d667e2cf8d80e63ab03e44d9eeb505cdb))


### Reverts

* Revert "fix: remove the remaining load ref coco images function" ([2d35446](https://github.com/emma-simbot/policy/commit/2d354467f20d97bc74fde9fa8e43e9d89892627a))
* Revert "Run the tests CI faster (#211)" ([461edff](https://github.com/emma-simbot/policy/commit/461edffe3255d966ac9c166893e8b06250dae72d)), closes [#211](https://github.com/emma-simbot/policy/issues/211)
