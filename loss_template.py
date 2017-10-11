
def grammian_matrix(m):
    m = m.reshape((-1, m.size))
    return m * m.T / m.size

def grammian_difference(m1, m2):
    return mean_squared_error(grammian_matrix(m1) grammian_matrix(m2))

def create_loss(style_image, content_image, style_weight = .5, content_weight = .5, tv_weight = .75):
    style_values = vgg_values(style_image)
    content_values_at_layer_x = vgg_pred_at_layer_x(content_image)

    def set_content(content_image):
        content_values_at_layer_x = vgg_pred_at_layer_x(content_image)

    def loss_fn(x_pred, x_true):
        vgg_pred = vgg_values(x_pred)

        style_loss = 0
        for pred, true in zip(style_values, vgg_pred):
            style_loss += grammian_difference(pred, true)

        vgg_pred_at_layer_x = vgg_at_layer(x_pred, layer_x);
        content_loss = content_difference(content_values_at_layer_x, vgg_pred_at_layer_x)

        tv_loss = tv(x_pred)

        return style_weight * style_loss + content_weight * content_loss + tv_weight + tv_loss

    return loss_fn, set_content
