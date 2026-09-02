# Exp87 result — SAH expert scope attribution

Decision: `close_independent_sah_expert_scope_branch`.

Exp87 tested whether Exp86 failed because too much of the decoder and BatchNorm
state were updated. Using patient-safe internal train folds 0/3 and internal dev
fold 4, it compared identical batches across head-only (`145` parameters), final
decoder block plus head (`7,121`) and full decoder (`2,836,401`). Normalization
running statistics were frozen. Calibration1 and outer2 were not inferred.

Head-only preserved ranking best, but its dev AP on background/IPH was `0.111027`
versus incumbent `0.122135`, and near-foreground AP was `0.121789` versus
`0.130836`. The final block reached `0.109449/0.115613`; the full decoder fell to
`0.085598/0.100803`. All three scopes failed the locked AP/separability gate.

The failure is therefore not explained only by BatchNorm drift. Increasing decoder
freedom worsened AP, while limiting the update still failed to beat the incumbent.
No checkpoint was saved. Runtime was `174.97s`, peak VRAM `1.223 GiB`, and exact
batch identity across scopes was verified by SHA256
`441af085eca3d51a786f80e2b18c137fc47e066c306055850312537b61e166ad`.
