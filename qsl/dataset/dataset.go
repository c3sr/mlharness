package dataset

var (
	defaultChannelBuffer = 100000
)

type Dataset interface {
	LoadQuerySamples([]int) error
	UnloadQuerySamples([]int) error
	GetSamples([]int) (map[int]interface{}, error)
	GetItemCount() int
}
